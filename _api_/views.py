from rest_framework import viewsets, generics, status, permissions
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.response import Response
from rest_framework.authtoken.models import Token
from rest_framework.permissions import IsAuthenticated, IsAdminUser, AllowAny
from django.contrib.auth import login, logout
from django.db.models import Sum, Count, Avg, Q
from django.utils import timezone
from datetime import timedelta
from decimal import Decimal
import uuid
from django.conf import settings
from .models import *
from .serializers import *
from django.contrib.auth.tokens import default_token_generator
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.utils.encoding import force_bytes, force_str
from django.contrib.auth.views import PasswordResetView as AuthPasswordResetView
from rest_framework.decorators import action
from django.core.mail import send_mail
from django.template.loader import render_to_string
from .ai_services import ai_service


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer
    # permission_classes = [IsAdminUser]

    @action(detail=False, methods=['get'], permission_classes=[IsAuthenticated])
    def profile(self, request):
        serializer = self.get_serializer(request.user)
        return Response(serializer.data)

    @action(detail=False, methods=['put'], permission_classes=[IsAuthenticated])
    def update_profile(self, request):
        user = request.user
        serializer = self.get_serializer(user, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class AuthViewSet(viewsets.ViewSet):
    permission_classes = [AllowAny]
    
    @action(detail=False, methods=['post'])
    def register(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            token = Token.objects.create(user=user)  # Create token for new user
            return Response({
                'user': UserSerializer(user).data,
                'token': token.key,
                'message': 'Registration successful'
            })
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    @action(detail=False, methods=['post'])
    def login(self, request):
        # Use DRF's ObtainAuthToken for proper token authentication
        username = request.data.get('username')
        password = request.data.get('password')
        
        if not username or not password:
            return Response(
                {'error': 'Please provide both username and password'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Authenticate user
        user = authenticate(username=username, password=password)
        
        if not user:
            return Response(
                {'error': 'Invalid credentials'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Get or create token
        token, created = Token.objects.get_or_create(user=user)
        
        return Response({
            'user': UserSerializer(user).data,
            'token': token.key,
            'message': 'Login successful'
        })
    
    @action(detail=False, methods=['post'], permission_classes=[IsAuthenticated])
    def logout(self, request):
        # Delete the token to logout
        Token.objects.filter(user=request.user).delete()
        return Response({'message': 'Logged out successfully'})

class CategoryViewSet(viewsets.ModelViewSet):
    queryset = Category.objects.all()
    serializer_class = CategorySerializer
    permission_classes = [IsAuthenticated]
    lookup_field = 'slug'

    def get_permissions(self):
        if self.action in ['list', 'retrieve']:
            return [AllowAny()]
        return [IsAdminUser()]

class TechnologyViewSet(viewsets.ModelViewSet):
    queryset = Technology.objects.all()
    serializer_class = TechnologySerializer
    permission_classes = [IsAdminUser]

class ProjectViewSet(viewsets.ModelViewSet):
    queryset = Project.objects.filter(is_active=True)
    serializer_class = ProjectSerializer
    lookup_field = 'slug'

    def get_permissions(self):
        # Public actions
        if self.action in ['list', 'retrieve', 'search']:
            return [AllowAny()]
        # Authenticated user actions
        elif self.action in ['add_to_cart', 'add_review']:
            return [IsAuthenticated()]
        # Admin only actions (create, update, destroy, etc.)
        else:
            return [IsAdminUser()]

    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        instance.views += 1
        instance.save(update_fields=['views'])
        
        # Track view for authenticated users
        if request.user.is_authenticated:
            ProjectView.objects.get_or_create(
                user=request.user,
                project=instance
            )
        
        serializer = self.get_serializer(instance)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def search(self, request):
        category_slug = request.GET.get('category', '')
        tech_slug = request.GET.get('tech', '')
        difficulty = request.GET.get('difficulty', '')
        search_query = request.GET.get('q', '')
        sort_by = request.GET.get('sort', 'newest')

        queryset = self.get_queryset()
        
        if category_slug:
            queryset = queryset.filter(category__slug=category_slug)
        
        if tech_slug:
            queryset = queryset.filter(technologies__slug=tech_slug)
        
        if difficulty:
            queryset = queryset.filter(difficulty=difficulty)
        
        if search_query:
            queryset = queryset.filter(
                Q(title__icontains=search_query) |
                Q(description__icontains=search_query) |
                Q(short_description__icontains=search_query) |
                Q(technologies__name__icontains=search_query)
            ).distinct()
        
        if sort_by == 'price_low':
            queryset = queryset.order_by('base_price')
        elif sort_by == 'price_high':
            queryset = queryset.order_by('-base_price')
        elif sort_by == 'popular':
            queryset = queryset.order_by('-purchases', '-views')
        elif sort_by == 'featured':
            queryset = queryset.filter(is_featured=True).order_by('-created_at')
        else:
            queryset = queryset.order_by('-created_at')

        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def add_to_cart(self, request, slug=None):
        project = self.get_object()
        cart, created = Cart.objects.get_or_create(user=request.user)
        custom_requirements = request.data.get('custom_requirements', '')
        
        cart_item, item_created = CartItem.objects.get_or_create(
            cart=cart,
            project=project,
            defaults={'custom_requirements': custom_requirements}
        )
        
        if not item_created:
            cart_item.custom_requirements = custom_requirements
            cart_item.save()
            message = f"{project.title} updated in cart"
        else:
            message = f"{project.title} added to cart"
        
        return Response({'message': message})

    # @action(detail=True, methods=['post'])
    # def add_review(self, request, slug=None):
    #     project = self.get_object()
        
    #     # Check if user has purchased this project
    #     has_purchased = Order.objects.filter(
    #         user=request.user, 
    #         project=project, 
    #         payment_status=True
    #     ).exists()
        
    #     if not has_purchased:
    #         return Response(
    #             {'error': 'You need to purchase this project before reviewing'}, 
    #             status=status.HTTP_400_BAD_REQUEST
    #         )
        
    #     # Check if user already reviewed
    #     existing_review = Review.objects.filter(project=project, user=request.user).first()
        
    #     if existing_review:
    #         serializer = ReviewSerializer(existing_review, data=request.data, partial=True)
    #     else:
    #         serializer = ReviewSerializer(data=request.data)
        
    #     if serializer.is_valid():
    #         review = serializer.save(project=project, user=request.user, is_approved=True)
    #         return Response(ReviewSerializer(review).data)
        
    #     return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    @action(detail=True, methods=['post'])
    def add_review(self, request, slug=None):
        project = self.get_object()

        # ✅ purchase check (keep here for clear API error)
        if not Order.objects.filter(
            user=request.user,
            project=project,
            payment_status=True
        ).exists():
            return Response(
                {"error": "You need to purchase this project before reviewing"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # ✅ check existing review
        existing_review = Review.objects.filter(
            project=project,
            user=request.user
        ).first()

        serializer = ReviewSerializer(
            instance=existing_review,
            data=request.data,
            partial=bool(existing_review),
            context={
                "request": request,
                "project": project  # ✅ This is the key fix!
            }
        )

        serializer.is_valid(raise_exception=True)

        review = serializer.save(
            user=request.user,
            project=project,
            is_approved=True
        )

        return Response(
            {
                "message": "Review submitted successfully",
                "review": ReviewSerializer(
                    review,
                    context={
                        "request": request,
                        "project": project
                    }
                ).data
            },
            status=status.HTTP_201_CREATED
        )

class ReviewViewSet(viewsets.ModelViewSet):
    queryset = Review.objects.filter(is_approved=True)
    serializer_class = ReviewSerializer

    def get_permissions(self):
        if self.action in ['list', 'retrieve']:
            return [AllowAny()]
        return [IsAuthenticated()]

    def perform_create(self, serializer):
        # Check if user has purchased the project
        project = serializer.validated_data['project']
        has_purchased = Order.objects.filter(
            user=self.request.user, 
            project=project, 
            payment_status=True
        ).exists()
        
        if not has_purchased:
            raise serializers.ValidationError(
                "You need to purchase this project before reviewing"
            )
        
        # Check if user already reviewed
        if Review.objects.filter(project=project, user=self.request.user).exists():
            raise serializers.ValidationError(
                "You have already reviewed this project"
            )
        
        serializer.save(user=self.request.user, is_approved=True)

    @action(detail=False, methods=['get'])
    def my_reviews(self, request):
        reviews = Review.objects.filter(user=request.user)
        serializer = self.get_serializer(reviews, many=True)
        return Response(serializer.data)
    
class CartViewSet(viewsets.ViewSet):
    permission_classes = [IsAuthenticated]
    
    def list(self, request):
        cart, created = Cart.objects.get_or_create(user=request.user)
        cart_items = cart.items.all()
        
        subtotal = sum(item.total_price for item in cart_items)
        if isinstance(subtotal, Decimal):
            gst = subtotal * Decimal('0.18')
        else:
            subtotal = Decimal(str(subtotal)) if subtotal else Decimal('0')
            gst = subtotal * Decimal('0.18')
        
        total = subtotal + gst
        
        # FIXED: Pass request context to serializer
        serializer = CartItemSerializer(
            cart_items, 
            many=True,
            context={'request': request}  # Pass context here
        )
        
        cart_serializer = CartSerializer(
            cart, 
            context={'request': request}
        )
        
        return Response({
            'cart': cart_serializer.data,
            'cart_items': serializer.data,
            'subtotal': subtotal,
            'gst': gst,
            'total': total,
            'total_items': cart.total_items
        })
    
    @action(detail=False, methods=['post'], url_path='add/(?P<project_id>[^/.]+)')
    def add_item(self, request, project_id=None):
        custom_requirements = request.data.get('custom_requirements', '')
        action_type = request.data.get('action_type', 'add_to_cart')
        
        try:
            project = Project.objects.get(id=project_id, is_active=True)
        except Project.DoesNotExist:
            return Response({'error': 'Project not found'}, status=404)
        
        cart, created = Cart.objects.get_or_create(user=request.user)
        cart_item, item_created = CartItem.objects.get_or_create(
            cart=cart,
            project=project,
            defaults={'custom_requirements': custom_requirements}
        )
        
        if not item_created:
            cart_item.custom_requirements = custom_requirements
            cart_item.save()
            message = f"{project.title} updated in cart"
        else:
            if action_type == 'buy_now':
                message = f"Processing purchase for {project.title}"
            else:
                message = f"{project.title} added to cart"
        
        response_data = {'message': message}
        
        # If buy_now, return checkout data
        if action_type == 'buy_now':
            # Create order directly
            order = Order.objects.create(
                user=request.user,
                project=project,
                amount=project.current_price,
                final_amount=project.current_price,
                custom_requirements=custom_requirements,
                status='pending'
            )
            
            # Remove from cart
            cart_item.delete()
            
            project.purchases += 1
            project.save(update_fields=['purchases'])
            
            response_data.update({
                'redirect': f'/api/orders/{order.order_id}/',
                'order_id': order.order_id,
                'order': OrderSerializer(order, context={'request': request}).data
            })
        else:
            # Return cart item data with context
            serializer = CartItemSerializer(
                cart_item, 
                context={'request': request}
            )
            response_data['cart_item'] = serializer.data
        
        return Response(response_data)
    
    @action(detail=False, methods=['delete'], url_path='remove/(?P<item_id>[^/.]+)')
    def remove_item(self, request, item_id=None):
        try:
            cart_item = CartItem.objects.get(id=item_id, cart__user=request.user)
            project_title = cart_item.project.title
            cart_item.delete()
            return Response({'message': f"{project_title} removed from cart"})
        except CartItem.DoesNotExist:
            return Response({'error': 'Cart item not found'}, status=404)
    
    @action(detail=False, methods=['put'], url_path='update/(?P<item_id>[^/.]+)')
    def update_item(self, request, item_id=None):
        try:
            cart_item = CartItem.objects.get(id=item_id, cart__user=request.user)
            custom_req = request.data.get('custom_requirements', '')
            cart_item.custom_requirements = custom_req
            cart_item.save()
            
            # Return updated cart item with context
            serializer = CartItemSerializer(
                cart_item, 
                context={'request': request}
            )
            return Response({
                'message': 'Custom requirements updated',
                'cart_item': serializer.data
            })
        except CartItem.DoesNotExist:
            return Response({'error': 'Cart item not found'}, status=404)
    
    @action(detail=False, methods=['delete'])
    def clear(self, request):
        cart, created = Cart.objects.get_or_create(user=request.user)
        cart.items.all().delete()
        return Response({'message': 'Cart cleared successfully'})

# class OrderViewSet(viewsets.ModelViewSet):
#     serializer_class = OrderSerializer
#     permission_classes = [IsAuthenticated]
#     lookup_field = 'order_id'  # Add this line
#     lookup_url_kwarg = 'order_id'  # Add this line

#     def get_queryset(self):
#         return Order.objects.filter(user=self.request.user)

#     @action(detail=False, methods=['post'])
#     def checkout(self, request):
#         try:
#             cart = Cart.objects.get(user=request.user)
#         except Cart.DoesNotExist:
#             return Response({'error': 'Cart is empty'}, status=400)
        
#         if cart.total_items == 0:
#             return Response({'error': 'Cart is empty'}, status=400)
        
#         orders = []
#         for cart_item in cart.items.all():
#             order = Order.objects.create(
#                 user=request.user,
#                 project=cart_item.project,
#                 amount=cart_item.project.current_price,
#                 final_amount=cart_item.project.current_price,
#                 custom_requirements=cart_item.custom_requirements,
#                 status='pending'
#             )
#             orders.append(order)
            
#             cart_item.project.purchases += 1
#             cart_item.project.save(update_fields=['purchases'])
        
#         cart.items.all().delete()
        
#         if len(orders) == 1:
#             serializer = self.get_serializer(orders[0])
#             return Response({
#                 'message': f"Order created successfully! Order ID: {orders[0].order_id}",
#                 'order': serializer.data
#             })
#         else:
#             serializer = self.get_serializer(orders, many=True)
#             return Response({
#                 'message': f"Successfully created {len(orders)} orders!",
#                 'orders': serializer.data
#             })

#     @action(detail=True, methods=['get'])
#     def download(self, request, pk=None):
#         order = self.get_object()
        
#         if not (order.payment_status and order.status == 'completed' and order.delivered_files):
#             return Response({'error': 'Files are not available for download yet'}, status=400)
        
#         order.download_count = getattr(order, 'download_count', 0) + 1
#         order.save(update_fields=['download_count'])
        
#         return Response({
#             'message': 'Download started',
#             'file_url': order.delivered_files.url
#         })

class OrderViewSet(viewsets.ModelViewSet):
    serializer_class = OrderSerializer
    permission_classes = [IsAuthenticated]
    lookup_field = 'order_id'
    lookup_url_kwarg = 'order_id'

    def get_queryset(self):
        return Order.objects.filter(user=self.request.user)

    @action(detail=False, methods=['post'])
    def checkout(self, request):
        try:
            cart = Cart.objects.get(user=request.user)
        except Cart.DoesNotExist:
            return Response({'error': 'Cart is empty'}, status=400)

        if cart.total_items == 0:
            return Response({'error': 'Cart is empty'}, status=400)

        orders = []
        for cart_item in cart.items.all():
            order = Order.objects.create(
                user=request.user,
                project=cart_item.project,
                amount=cart_item.project.current_price,
                final_amount=cart_item.project.current_price,
                custom_requirements=cart_item.custom_requirements,
                status='pending'
            )
            orders.append(order)

            cart_item.project.purchases += 1
            cart_item.project.save(update_fields=['purchases'])

        cart.items.all().delete()

        if len(orders) == 1:
            serializer = self.get_serializer(orders[0])
            return Response({
                'message': f"Order created successfully! Order ID: {orders[0].order_id}",
                'order': serializer.data
            })

        serializer = self.get_serializer(orders, many=True)
        return Response({
            'message': f"Successfully created {len(orders)} orders!",
            'orders': serializer.data
        })

    @action(detail=True, methods=['get'])
    def download(self, request, order_id=None):
        order = self.get_object()

        if not (order.payment_status and order.status == 'completed' and order.delivered_files):
            return Response(
                {'error': 'Files are not available for download yet'},
                status=400
            )

        order.download_count = getattr(order, 'download_count', 0) + 1
        order.save(update_fields=['download_count'])

        return Response({
            'message': 'Download started',
            'file_url': order.delivered_files.url
        })


class ReferralViewSet(viewsets.ViewSet):
    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=['get'])
    def dashboard(self, request):
        referrals_made = Referral.objects.filter(referrer=request.user)
        total_earned = referrals_made.filter(credited=True).aggregate(
            total=Sum('bonus_amount')
        )['total'] or 0
        
        referral_url = f"{settings.SITE_URL}/api/auth/register/?ref={request.user.referral_code}"
        
        serializer = ReferralSerializer(referrals_made, many=True)
        return Response({
            'referrals': serializer.data,
            'total_earned': total_earned,
            'referral_url': referral_url,
            'referral_code': request.user.referral_code,
            'pending_referrals': referrals_made.filter(credited=False).count(),
        })

class DashboardViewSet(viewsets.ViewSet):
    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=['get'])
    def stats(self, request):
        user = request.user
        today = timezone.now()
        
        # Get statistics
        total_orders = Order.objects.filter(user=user, payment_status=True).count()
        completed_orders = Order.objects.filter(user=user, status='completed', payment_status=True).count()
        pending_orders = Order.objects.filter(
            user=user, 
            status__in=['pending', 'processing'], 
            payment_status=True
        ).count()
        
        total_spent_result = Order.objects.filter(
            user=user, 
            payment_status=True
        ).aggregate(total=Sum('final_amount'))
        total_spent = total_spent_result['total'] or 0
        
        avg_order_value = total_spent / total_orders if total_orders > 0 else 0
        completion_rate = (completed_orders / total_orders * 100) if total_orders > 0 else 0
        
        month_start = today.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        monthly_spending_result = Order.objects.filter(
            user=user,
            payment_status=True,
            created_at__gte=month_start
        ).aggregate(total=Sum('final_amount'))
        monthly_spending = monthly_spending_result['total'] or 0
        
        avg_rating_result = Review.objects.filter(
            user=user,
            is_approved=True
        ).aggregate(avg_rating=Avg('rating'))
        avg_project_rating = avg_rating_result['avg_rating'] or 0
        
        week_ago = today - timedelta(days=7)
        recent_activity = Order.objects.filter(
            user=user,
            created_at__gte=week_ago
        ).count()
        
        # Get recent orders
        recent_orders = Order.objects.filter(user=user).select_related('project').order_by('-created_at')[:10]
        orders_serializer = OrderSerializer(recent_orders, many=True)
        
        # Get recent views
        recent_views = ProjectView.objects.filter(user=user).select_related('project').order_by('-viewed_at')[:6]
        views_serializer = ProjectViewSerializer(recent_views, many=True)
        
        # Get recommended projects
        recommended_projects = self.get_recommended_projects(user)
        projects_serializer = ProjectSerializer(recommended_projects, many=True, context={'request': request})
        
        # Get favorite category
        favorite_category = self.get_favorite_category(user)
        
        return Response({
            'stats': {
                'total_orders': total_orders,
                'completed_orders': completed_orders,
                'pending_orders': pending_orders,
                'total_spent': total_spent,
                'avg_order_value': avg_order_value,
                'completion_rate': round(completion_rate),
                'monthly_spending': monthly_spending,
                'avg_project_rating': avg_project_rating,
                'recent_activity': recent_activity,
            },
            'recent_orders': orders_serializer.data,
            'recent_views': views_serializer.data,
            'recommended_projects': projects_serializer.data,
            'favorite_category': favorite_category,
        })

    def get_recommended_projects(self, user):
        purchased_categories = Order.objects.filter(
            user=user,
            payment_status=True
        ).values_list('project__category', flat=True).distinct()
        
        viewed_categories = ProjectView.objects.filter(
            user=user
        ).values_list('project__category', flat=True).distinct()
        
        all_categories = list(purchased_categories) + list(viewed_categories)
        
        if all_categories:
            purchased_project_ids = Order.objects.filter(
                user=user,
                payment_status=True
            ).values_list('project__id', flat=True)
            
            recommended = Project.objects.filter(
                category_id__in=all_categories,
                is_active=True
            ).exclude(
                id__in=purchased_project_ids
            ).order_by(
                '-is_featured', 
                '-purchases', 
                '-views'
            )[:6]
        else:
            recommended = Project.objects.filter(
                is_active=True,
                is_featured=True
            ).order_by('-purchases', '-views')[:6]
        
        return recommended

    def get_favorite_category(self, user):
        category_stats = Order.objects.filter(
            user=user,
            payment_status=True
        ).values(
            'project__category__name',
            'project__category__slug'
        ).annotate(
            count=Count('id')
        ).order_by('-count')
        
        if category_stats:
            return {
                'name': category_stats[0]['project__category__name'],
                'slug': category_stats[0]['project__category__slug'],
                'count': category_stats[0]['count']
            }
        return None

class AdminViewSet(viewsets.ViewSet):
    permission_classes = [IsAdminUser]

    @action(detail=False, methods=['get'])
    def dashboard(self, request):
        today = timezone.now().date()
        week_ago = today - timedelta(days=7)
        month_ago = today - timedelta(days=30)
        
        # User statistics
        total_users = User.objects.count()
        new_users_today = User.objects.filter(date_joined__date=today).count()
        new_users_week = User.objects.filter(date_joined__date__gte=week_ago).count()
        
        # Project statistics
        total_projects = Project.objects.count()
        active_projects = Project.objects.filter(is_active=True).count()
        featured_projects = Project.objects.filter(is_featured=True).count()
        
        # Order statistics
        total_orders = Order.objects.count()
        completed_orders = Order.objects.filter(status='completed').count()
        pending_orders = Order.objects.filter(status='pending').count()
        
        # Revenue statistics
        total_revenue = Order.objects.filter(payment_status=True).aggregate(
            Sum('final_amount')
        )['final_amount__sum'] or 0
        
        today_revenue = Order.objects.filter(
            created_at__date=today,
            payment_status=True
        ).aggregate(Sum('final_amount'))['final_amount__sum'] or 0
        
        week_revenue = Order.objects.filter(
            created_at__date__gte=week_ago,
            payment_status=True
        ).aggregate(Sum('final_amount'))['final_amount__sum'] or 0
        
        # Recent activities
        recent_orders = Order.objects.select_related('user', 'project').order_by('-created_at')[:10]
        recent_users = User.objects.order_by('-date_joined')[:10]
        
        # Top selling projects
        top_projects = Project.objects.annotate(
            total_sales=Count('orders')
        ).order_by('-total_sales', '-purchases')[:5]
        
        # Category distribution
        categories = Category.objects.annotate(
            project_count=Count('projects'),
            sales_count=Count('projects__orders')
        ).order_by('-sales_count')[:5]
        
        orders_serializer = OrderSerializer(recent_orders, many=True)
        users_serializer = UserSerializer(recent_users, many=True)
        projects_serializer = ProjectSerializer(top_projects, many=True)
        categories_serializer = CategorySerializer(categories, many=True)
        
        return Response({
            'stats': {
                'total_users': total_users,
                'new_users_today': new_users_today,
                'new_users_week': new_users_week,
                'total_projects': total_projects,
                'active_projects': active_projects,
                'featured_projects': featured_projects,
                'total_orders': total_orders,
                'completed_orders': completed_orders,
                'pending_orders': pending_orders,
                'total_revenue': total_revenue,
                'today_revenue': today_revenue,
                'week_revenue': week_revenue,
            },
            'recent_orders': orders_serializer.data,
            'recent_users': users_serializer.data,
            'top_projects': projects_serializer.data,
            'categories': categories_serializer.data,
        })
    
    @action(detail=False, methods=['post'], url_path='orders/update/(?P<order_id>[^/.]+)')
    def update_order(self, request, order_id=None):
        try:
            order = Order.objects.get(order_id=order_id)
        except Order.DoesNotExist:
            return Response({'error': 'Order not found'}, status=404)
        
        status = request.data.get('status')
        delivery_message = request.data.get('delivery_message', '')
        files = request.FILES.get('delivered_files')
        
        old_status = order.status
        order.status = status
        order.delivery_message = delivery_message
        
        if files:
            order.delivered_files = files
        
        if status == 'completed' and old_status != 'completed':
            order.completed_at = timezone.now()
            # Send completion email (you'll need to implement this)
        
        order.save()
        
        return Response({
            'message': f"Order {order_id} updated successfully",
            'order': OrderSerializer(order).data
        })

    @action(detail=False, methods=['get'])
    def orders(self, request):
        orders = Order.objects.select_related('user', 'project').order_by('-created_at')
        
        status_filter = request.GET.get('status', '')
        payment_filter = request.GET.get('payment', '')
        
        if status_filter:
            orders = orders.filter(status=status_filter)
        if payment_filter:
            orders = orders.filter(payment_status=(payment_filter == 'paid'))
        
        serializer = OrderSerializer(orders, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def users(self, request):
        users = User.objects.all().order_by('-date_joined')
        
        college_filter = request.GET.get('college', '')
        if college_filter:
            users = users.filter(college__icontains=college_filter)
        
        serializer = UserSerializer(users, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['get'])
    def projects(self, request):
        projects = Project.objects.all().order_by('-created_at')
        
        category_filter = request.GET.get('category', '')
        active_filter = request.GET.get('active', '')
        
        if category_filter:
            projects = projects.filter(category__slug=category_filter)
        if active_filter:
            projects = projects.filter(is_active=(active_filter == 'true'))
        
        serializer = ProjectSerializer(projects, many=True)
        return Response(serializer.data)

@api_view(['GET'])
@permission_classes([AllowAny])
def home(request):
    return Response({
        'message': 'Welcome to Codora API',
        'endpoints': {
            'auth': {
                'register': '/api/auth/register/',
                'login': '/api/auth/login/',
                'logout': '/api/auth/logout/',
            },
            'projects': '/api/projects/',
            'dashboard': '/api/dashboard/stats/',
            'cart': '/api/cart/',
            'orders': '/api/orders/',
        }
    })
    

# Add this new ViewSet for password reset
class PasswordResetViewSet(viewsets.ViewSet):
    permission_classes = [AllowAny]
    
    @action(detail=False, methods=['post'])
    def password_reset(self, request):
        email = request.data.get('email')
        
        if not email:
            return Response({'error': 'Email is required'}, status=400)
        
        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            # For security, don't reveal if email exists or not
            return Response({
                'message': 'If an account exists with this email, you will receive a password reset link.'
            })
        
        # Generate token and uid
        token = default_token_generator.make_token(user)
        uid = urlsafe_base64_encode(force_bytes(user.pk))
        
        # Create reset URL
        reset_url = f"{settings.SITE_URL}/api/accounts/password-reset-confirm/{uid}/{token}/"
        
        # Send email
        subject = "Password Reset Request - Codora"
        message = f"""
        Hello {user.username},
        
        You requested a password reset for your Codora account.
        
        Please click the link below to reset your password:
        {reset_url}
        
        This link will expire in 24 hours.
        
        If you didn't request this, please ignore this email.
        
        Best regards,
        Codora Team
        """
        
        try:
            send_mail(
                subject=subject,
                message=message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[user.email],
                fail_silently=False,
            )
            return Response({
                'message': 'Password reset email sent successfully',
                'uid': uid,
                'token': token,
                'reset_url': reset_url  # For testing only, remove in production
            })
        except Exception as e:
            return Response({
                'error': 'Failed to send email',
                'detail': str(e)
            }, status=500)
    
    @action(detail=False, methods=['get'])
    def password_reset_done(self, request):
        return Response({
            'message': 'Password reset email sent successfully. Please check your email.'
        })
    
    @action(detail=False, methods=['post'], url_path='password-reset-confirm/(?P<uidb64>[^/.]+)/(?P<token>[^/.]+)')
    def password_reset_confirm(self, request, uidb64=None, token=None):
        try:
            uid = force_str(urlsafe_base64_decode(uidb64))
            user = User.objects.get(pk=uid)
        except (TypeError, ValueError, OverflowError, User.DoesNotExist):
            user = None
        
        if user is not None and default_token_generator.check_token(user, token):
            password = request.data.get('password')
            password2 = request.data.get('password2')
            
            if not password or not password2:
                return Response({'error': 'Both password fields are required'}, status=400)
            
            if password != password2:
                return Response({'error': 'Passwords do not match'}, status=400)
            
            # Validate password
            from django.contrib.auth.password_validation import validate_password
            from django.core.exceptions import ValidationError
            
            try:
                validate_password(password, user)
            except ValidationError as e:
                return Response({'error': 'Password validation failed', 'details': list(e.messages)}, status=400)
            
            # Set new password
            user.set_password(password)
            user.save()
            
            return Response({'message': 'Password reset successful. You can now login with your new password.'})
        
        return Response({'error': 'Invalid or expired reset link'}, status=400)
    
    @action(detail=False, methods=['get'])
    def password_reset_complete(self, request):
        return Response({
            'message': 'Password reset completed successfully. You can now login with your new password.'
        })

# Add this new ViewSet for project preview
class ProjectPreviewViewSet(viewsets.ViewSet):
    permission_classes = [AllowAny]
    
    @action(detail=False, methods=['get'], url_path='(?P<project_slug>[^/.]+)')
    def preview(self, request, project_slug=None):
        try:
            project = Project.objects.get(slug=project_slug, is_active=True)
            can_view_full = False
            
            if request.user.is_authenticated:
                can_view_full = Order.objects.filter(
                    user=request.user, 
                    project=project, 
                    payment_status=True
                ).exists()
            
            serializer = ProjectSerializer(project, context={'request': request})
            return Response({
                'project': serializer.data,
                'can_view_full': can_view_full,
                'has_preview': hasattr(project, 'preview') and project.preview
            })
        except Project.DoesNotExist:
            return Response({'error': 'Project not found'}, status=404)
    
    @action(detail=False, methods=['get'], url_path='(?P<project_slug>[^/.]+)/demo')
    def live_demo(self, request, project_slug=None):
        try:
            project = Project.objects.get(slug=project_slug, is_active=True)
            
            # Check if preview exists
            if not hasattr(project, 'preview') or not project.preview.live_demo_url:
                return Response({'error': 'Live demo not available'}, status=404)
            
            # Check access for authenticated users
            if request.user.is_authenticated:
                has_access = Order.objects.filter(
                    user=request.user, 
                    project=project, 
                    payment_status=True
                ).exists()
                
                if not has_access:
                    return Response({
                        'error': 'Purchase this project to access live demo',
                        'redirect': f'/api/projects/{project_slug}/'
                    }, status=403)
            
            return Response({
                'demo_url': project.preview.live_demo_url,
                'project': ProjectSerializer(project, context={'request': request}).data
            })
        except Project.DoesNotExist:
            return Response({'error': 'Project not found'}, status=404)
        

#Ai Services ViewSet

class AIRecommendationViewSet(viewsets.ViewSet):
    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=['get'])
    def get_recommendations(self, request):
        """Get personalized project recommendations"""
        limit = int(request.GET.get('limit', 10))
        refresh = request.GET.get('refresh', 'false').lower() == 'true'
        
        if refresh:
            # Clear old recommendations
            Recommendation.objects.filter(user=request.user).delete()
        
        # Get recommendations
        recommendations = Recommendation.objects.filter(
            user=request.user,
            is_active=True,
            expires_at__gte=timezone.now()
        ).order_by('-final_score', 'position')[:limit]
        
        if not recommendations or refresh:
            # Generate new recommendations
            recommendations = ai_service.generate_recommendations(request.user, limit)
        
        serializer = RecommendationSerializer(recommendations, many=True)
        
        # Get recommendation reasons grouped by type
        by_type = {}
        for rec in recommendations:
            rec_type = rec.recommendation_type
            if rec_type not in by_type:
                by_type[rec_type] = []
            by_type[rec_type].append({
                'project_id': rec.project.id,
                'project_title': rec.project.title,
                'score': rec.final_score,
                'reason': rec.reason
            })
        
        return Response({
            'recommendations': serializer.data,
            'grouped_by_type': by_type,
            'total': len(recommendations)
        })

    @action(detail=False, methods=['post'])
    def generate(self, request):
        """Generate new recommendations"""
        serializer = AIRecommendationRequestSerializer(data=request.data)
        if serializer.is_valid():
            limit = serializer.validated_data['limit']
            
            # Clear old recommendations
            Recommendation.objects.filter(user=request.user).delete()
            
            # Generate new recommendations
            recommendations = ai_service.generate_recommendations(request.user, limit)
            
            serializer = RecommendationSerializer(recommendations, many=True)
            return Response({
                'message': f'Generated {len(recommendations)} recommendations',
                'recommendations': serializer.data
            })
        return Response(serializer.errors, status=400)

    @action(detail=False, methods=['post'])
    def feedback(self, request):
        """Provide feedback on recommendations"""
        project_id = request.data.get('project_id')
        action = request.data.get('action')  # 'click', 'purchase', 'dismiss'
        
        if not project_id or not action:
            return Response({'error': 'project_id and action are required'}, status=400)
        
        try:
            project = Project.objects.get(id=project_id)
            recommendation = Recommendation.objects.get(user=request.user, project=project)
            
            # Log the feedback
            UserBehavior.objects.create(
                user=request.user,
                action=f'recommendation_{action}',
                project=project,
                metadata={'recommendation_id': recommendation.id}
            )
            
            # Adjust recommendation score based on feedback
            if action == 'dismiss':
                recommendation.is_active = False
                recommendation.save()
            
            return Response({'message': f'Feedback recorded for {project.title}'})
        except (Project.DoesNotExist, Recommendation.DoesNotExist):
            return Response({'error': 'Recommendation not found'}, status=404)

class AISearchViewSet(viewsets.ViewSet):
    permission_classes = [AllowAny]

    @action(detail=False, methods=['post'])
    def intelligent_search(self, request):
        """AI-powered intelligent search"""
        serializer = AISearchRequestSerializer(data=request.data)
        if serializer.is_valid():
            query = serializer.validated_data['query']
            limit = serializer.validated_data['limit']
            
            filters = {}
            if serializer.validated_data.get('category'):
                filters['category'] = serializer.validated_data['category']
            if serializer.validated_data.get('difficulty'):
                filters['difficulty'] = serializer.validated_data['difficulty']
            if serializer.validated_data.get('min_price'):
                filters['min_price'] = serializer.validated_data['min_price']
            if serializer.validated_data.get('max_price'):
                filters['max_price'] = serializer.validated_data['max_price']
            
            # Perform intelligent search
            user = request.user if request.user.is_authenticated else None
            projects, intent = ai_service.intelligent_search(query, user, filters, limit)
            
            serializer = ProjectSerializer(projects, many=True, context={'request': request})
            
            return Response({
                'query': query,
                'detected_intent': intent,
                'results_count': len(projects),
                'results': serializer.data,
                'filters_applied': filters
            })
        return Response(serializer.errors, status=400)

    @action(detail=False, methods=['get'], permission_classes=[IsAuthenticated])
    def search_history(self, request):
        """Get user's search history"""
        searches = SearchHistory.objects.filter(user=request.user).order_by('-created_at')[:50]
        serializer = SearchHistorySerializer(searches, many=True)
        return Response(serializer.data)

class AILearningPathViewSet(viewsets.ViewSet):
    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=['post'])
    def generate_path(self, request):
        """Generate personalized learning path"""
        serializer = AILearningPathRequestSerializer(data=request.data)
        if serializer.is_valid():
            goal = serializer.validated_data.get('goal')
            
            # Generate learning path
            learning_path = ai_service.generate_learning_path(request.user, goal)
            
            if learning_path:
                serializer = LearningPathSerializer(learning_path)
                return Response({
                    'message': 'Learning path generated successfully',
                    'learning_path': serializer.data
                })
            else:
                return Response({
                    'error': 'Could not generate learning path. Try purchasing some projects first.'
                }, status=400)
        return Response(serializer.errors, status=400)

    @action(detail=False, methods=['get'])
    def my_paths(self, request):
        """Get user's learning paths"""
        paths = LearningPath.objects.filter(user=request.user, is_active=True)
        serializer = LearningPathSerializer(paths, many=True)
        return Response(serializer.data)

    @action(detail=True, methods=['post'])
    def update_progress(self, request, pk=None):
        """Update progress on a learning path"""
        try:
            learning_path = LearningPath.objects.get(id=pk, user=request.user)
            
            project_id = request.data.get('project_id')
            completed = request.data.get('completed', False)
            
            if project_id:
                # Update specific project completion
                try:
                    path_project = LearningPathProject.objects.get(
                        learning_path=learning_path,
                        project_id=project_id
                    )
                    path_project.is_completed = completed
                    if completed:
                        path_project.completed_at = timezone.now()
                    path_project.save()
                except LearningPathProject.DoesNotExist:
                    return Response({'error': 'Project not in learning path'}, status=404)
            
            # Recalculate overall progress
            total_projects = learning_path.learningpathproject_set.count()
            completed_projects = learning_path.learningpathproject_set.filter(is_completed=True).count()
            
            if total_projects > 0:
                learning_path.progress = completed_projects / total_projects
                learning_path.completed = learning_path.progress >= 0.99
                learning_path.save()
            
            serializer = LearningPathSerializer(learning_path)
            return Response({
                'message': 'Progress updated',
                'learning_path': serializer.data
            })
        except LearningPath.DoesNotExist:
            return Response({'error': 'Learning path not found'}, status=404)

class AIChatViewSet(viewsets.ViewSet):
    permission_classes = [IsAuthenticated]

    @action(detail=False, methods=['post'])
    def chat(self, request):
        """Chat with AI assistant"""
        serializer = AIChatRequestSerializer(data=request.data)
        if serializer.is_valid():
            message = serializer.validated_data['message']
            session_id = serializer.validated_data.get('session_id')
            context = serializer.validated_data.get('context', {})
            
            # Get or create session
            if session_id:
                try:
                    session = AIChatSession.objects.get(
                        session_id=session_id,
                        user=request.user,
                        is_active=True
                    )
                except AIChatSession.DoesNotExist:
                    session = AIChatSession.objects.create(
                        user=request.user,
                        session_id=session_id,
                        context=context
                    )
            else:
                session_id = str(uuid.uuid4())
                session = AIChatSession.objects.create(
                    user=request.user,
                    session_id=session_id,
                    context=context
                )
            
            # Save user message
            user_message = AIChatMessage.objects.create(
                session=session,
                message_type='user',
                content=message
            )
            
            # Generate AI response (simplified - integrate with actual AI model)
            ai_response = self._generate_ai_response(session, message)
            
            # Save AI response
            ai_message = AIChatMessage.objects.create(
                session=session,
                message_type='ai',
                content=ai_response['response'],
                metadata=ai_response.get('metadata', {})
            )
            
            # Update session context
            session.context.update(ai_response.get('context_updates', {}))
            session.detected_intent = ai_response.get('intent', '')
            session.save()
            
            # Add recommended projects if any
            if ai_response.get('recommended_projects'):
                project_ids = ai_response['recommended_projects']
                projects = Project.objects.filter(id__in=project_ids)
                session.recommended_projects.add(*projects)
            
            session_serializer = AIChatSessionSerializer(session)
            return Response({
                'session': session_serializer.data,
                'response': ai_response['response'],
                'session_id': session.session_id
            })
        return Response(serializer.errors, status=400)

    def _generate_ai_response(self, session, message):
        """Generate AI response (simplified - integrate with GPT/LLM)"""
        message_lower = message.lower()
        
        # Simple rule-based responses
        if any(word in message_lower for word in ['hello', 'hi', 'hey']):
            response = "Hello! I'm Codora AI assistant. How can I help you with projects today?"
            intent = 'greeting'
        elif any(word in message_lower for word in ['recommend', 'suggest', 'what should']):
            # Get recommendations
            recommendations = ai_service.generate_recommendations(session.user, 3)
            project_names = [rec.project.title for rec in recommendations[:3]]
            
            response = f"Based on your interests, I recommend: {', '.join(project_names)}. Would you like more details about any of these?"
            intent = 'recommendation'
            metadata = {
                'recommended_projects': [rec.project.id for rec in recommendations[:3]]
            }
        elif any(word in message_lower for word in ['help', 'guide', 'how to']):
            response = "I can help you with: finding projects, generating learning paths, getting recommendations, and answering questions about coding projects. What would you like help with?"
            intent = 'help'
        elif any(word in message_lower for word in ['price', 'cost', 'expensive']):
            response = "Project prices range from ₹999 to ₹9999 based on complexity. I can help you find projects within your budget. What's your budget range?"
            intent = 'pricing'
        else:
            response = "I understand you're asking about projects. Could you be more specific about what you're looking for? For example, you can ask for recommendations, help with a specific technology, or guidance on learning paths."
            intent = 'general'
        
        return {
            'response': response,
            'intent': intent,
            'metadata': metadata if 'metadata' in locals() else {},
            'context_updates': {'last_intent': intent}
        }

    @action(detail=False, methods=['get'])
    def sessions(self, request):
        """Get user's chat sessions"""
        sessions = AIChatSession.objects.filter(user=request.user, is_active=True)
        serializer = AIChatSessionSerializer(sessions, many=True)
        return Response(serializer.data)

    @action(detail=False, methods=['post'])
    def end_session(self, request):
        """End a chat session"""
        session_id = request.data.get('session_id')
        if not session_id:
            return Response({'error': 'session_id is required'}, status=400)
        
        try:
            session = AIChatSession.objects.get(
                session_id=session_id,
                user=request.user
            )
            session.is_active = False
            session.summary = "Session ended by user"
            session.save()
            return Response({'message': 'Session ended'})
        except AIChatSession.DoesNotExist:
            return Response({'error': 'Session not found'}, status=404)

class AIAnalyticsViewSet(viewsets.ViewSet):
    permission_classes = [IsAdminUser]

    @action(detail=False, methods=['get'])
    def user_behavior(self, request):
        """Get user behavior analytics"""
        # Recent behaviors
        recent_behaviors = UserBehavior.objects.all().order_by('-created_at')[:100]
        serializer = UserBehaviorSerializer(recent_behaviors, many=True)
        
        # Statistics
        total_behaviors = UserBehavior.objects.count()
        by_action = UserBehavior.objects.values('action').annotate(count=Count('id'))
        
        # Popular projects
        popular_projects = Project.objects.annotate(
            view_count=Count('projectview'),
            purchase_count=Count('orders', filter=Q(orders__payment_status=True))
        ).order_by('-view_count', '-purchase_count')[:10]
        
        return Response({
            'recent_behaviors': serializer.data,
            'statistics': {
                'total_behaviors': total_behaviors,
                'by_action': list(by_action)
            },
            'popular_projects': ProjectSerializer(popular_projects, many=True).data
        })

    @action(detail=False, methods=['get'])
    def anomalies(self, request):
        """Get detected anomalies"""
        anomalies = AnomalyDetection.objects.filter(is_resolved=False)
        serializer = AnomalyDetectionSerializer(anomalies, many=True)
        
        # Run new detection
        new_anomalies = ai_service.detect_anomalies()
        for anomaly in new_anomalies:
            AnomalyDetection.objects.create(
                anomaly_type=anomaly['type'],
                description=anomaly['description'],
                confidence_score=anomaly['confidence'],
                metadata=anomaly
            )
        
        return Response({
            'anomalies': serializer.data,
            'new_detections': len(new_anomalies)
        })

    @action(detail=False, methods=['post'])
    def resolve_anomaly(self, request):
        """Resolve an anomaly"""
        anomaly_id = request.data.get('anomaly_id')
        action = request.data.get('action', '')
        
        if not anomaly_id:
            return Response({'error': 'anomaly_id is required'}, status=400)
        
        try:
            anomaly = AnomalyDetection.objects.get(id=anomaly_id)
            anomaly.is_resolved = True
            anomaly.action_taken = action
            anomaly.resolved_by = request.user
            anomaly.resolved_at = timezone.now()
            anomaly.save()
            
            return Response({'message': 'Anomaly resolved'})
        except AnomalyDetection.DoesNotExist:
            return Response({'error': 'Anomaly not found'}, status=404)

    @action(detail=False, methods=['get'])
    def recommendations_analytics(self, request):
        """Get recommendation system analytics"""
        # Recommendation effectiveness
        recommendations = Recommendation.objects.all()
        total_recommendations = recommendations.count()
        active_recommendations = recommendations.filter(is_active=True).count()
        
        # Click-through rate (simulated)
        clicked_recommendations = UserBehavior.objects.filter(
            action__startswith='recommendation_click'
        ).count()
        
        ctr = (clicked_recommendations / total_recommendations * 100) if total_recommendations > 0 else 0
        
        # By type
        by_type = recommendations.values('recommendation_type').annotate(
            count=Count('id'),
            avg_score=Avg('final_score')
        )
        
        return Response({
            'total_recommendations': total_recommendations,
            'active_recommendations': active_recommendations,
            'click_through_rate': round(ctr, 2),
            'by_type': list(by_type)
        })