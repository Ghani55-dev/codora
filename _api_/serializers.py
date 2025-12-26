from rest_framework import serializers
from django.contrib.auth import authenticate
from django.contrib.auth.password_validation import validate_password
from .models import *
import uuid

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'phone', 
                 'college', 'year', 'course', 'wallet_balance', 'profile_image',
                 'referral_code', 'total_referrals', 'referral_earnings']
        read_only_fields = ['wallet_balance', 'referral_code', 'total_referrals', 'referral_earnings']

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, validators=[validate_password])
    password2 = serializers.CharField(write_only=True, required=True)
    ref_code = serializers.CharField(write_only=True, required=False, allow_blank=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'password2', 'first_name', 
                 'last_name', 'phone', 'college', 'year', 'course', 'ref_code']

    def validate(self, attrs):
        if attrs['password'] != attrs['password2']:
            raise serializers.ValidationError({"password": "Password fields didn't match."})
        return attrs

    def create(self, validated_data):
        ref_code = validated_data.pop('ref_code', None)
        password2 = validated_data.pop('password2', None)
        
        user = User.objects.create(
            referral_code=str(uuid.uuid4())[:8].upper(),
            **validated_data
        )
        user.set_password(validated_data['password'])
        
        # Handle referral
        if ref_code:
            try:
                referrer = User.objects.get(referral_code=ref_code)
                user.referred_by = referrer
                Referral.objects.create(
                    referrer=referrer,
                    referred_user=user,
                    bonus_amount=100
                )
            except User.DoesNotExist:
                pass
        
        user.save()
        return user

class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)

    def validate(self, attrs):
        user = authenticate(username=attrs['username'], password=attrs['password'])
        if not user:
            raise serializers.ValidationError("Invalid credentials")
        attrs['user'] = user
        return attrs

class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = '__all__'

class TechnologySerializer(serializers.ModelSerializer):
    class Meta:
        model = Technology
        fields = '__all__'

class ProjectSerializer(serializers.ModelSerializer):
    category = CategorySerializer(read_only=True)
    category_id = serializers.PrimaryKeyRelatedField(
        queryset=Category.objects.all(), 
        source='category',
        write_only=True
    )
    technologies = TechnologySerializer(many=True, read_only=True)
    technology_ids = serializers.PrimaryKeyRelatedField(
        queryset=Technology.objects.all(),
        many=True,
        source='technologies',
        write_only=True
    )
    features_list = serializers.SerializerMethodField()
    learning_outcomes_list = serializers.SerializerMethodField()
    current_price = serializers.SerializerMethodField()
    has_purchased = serializers.SerializerMethodField()

    class Meta:
        model = Project
        fields = '__all__'
        read_only_fields = ['views', 'purchases', 'created_at', 'updated_at']

    def get_features_list(self, obj):
        return obj.features_list

    def get_learning_outcomes_list(self, obj):
        return obj.learning_outcomes_list
    
    def get_current_price(self, obj):
        return obj.current_price

    def get_has_purchased(self, obj):
        # FIXED: Safe check for request context
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            try:
                return Order.objects.filter(
                    user=request.user, 
                    project=obj, 
                    payment_status=True
                ).exists()
            except:
                return False
        return False

# class ReviewSerializer(serializers.ModelSerializer):
#     user = UserSerializer(read_only=True)
#     user_id = serializers.PrimaryKeyRelatedField(
#         queryset=User.objects.all(),
#         source='user',
#         write_only=True
#     )
#     project_id = serializers.PrimaryKeyRelatedField(
#         queryset=Project.objects.all(),
#         source='project',
#         write_only=True
#     )
#     stars = serializers.SerializerMethodField()

#     class Meta:
#         model = Review
#         fields = '__all__'
#         read_only_fields = ['created_at', 'updated_at']

#     def get_stars(self, obj):
#         return '★' * obj.rating + '☆' * (5 - obj.rating)

#     def validate(self, attrs):
#         user = attrs['user']
#         project = attrs['project']
        
#         # Check if user has purchased the project
#         if not Order.objects.filter(user=user, project=project, payment_status=True).exists():
#             raise serializers.ValidationError("You need to purchase this project before reviewing")
        
#         # Check if user already reviewed
#         if Review.objects.filter(user=user, project=project).exists():
#             raise serializers.ValidationError("You have already reviewed this project")
        
#         return attrs

class ReviewSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    project = ProjectSerializer(read_only=True)
    project_id = serializers.PrimaryKeyRelatedField(
        queryset=Project.objects.filter(is_active=True),
        write_only=True,
        source='project',
        required=False  # Make optional for add_review endpoint
    )
    stars = serializers.SerializerMethodField()
    user_has_purchased = serializers.SerializerMethodField()

    class Meta:
        model = Review
        fields = [
            'id', 'user', 'project', 'project_id', 
            'rating', 'stars', 'comment', 'is_approved',
            'created_at', 'updated_at', 'user_has_purchased'
        ]
        read_only_fields = [
            'user', 'project', 'is_approved', 
            'created_at', 'updated_at', 'stars', 'user_has_purchased'
        ]

    def get_stars(self, obj):
        return '★' * obj.rating + '☆' * (5 - obj.rating)

    def get_user_has_purchased(self, obj):
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            return Order.objects.filter(
                user=request.user,
                project=obj.project,
                payment_status=True
            ).exists()
        return False

    def validate(self, attrs):
        rating = attrs.get('rating')
        if rating < 1 or rating > 5:
            raise serializers.ValidationError({
                "rating": "Rating must be between 1 and 5"
            })
        return attrs
    
    def create(self, validated_data):
        """
        Override create to handle project from context
        """
        # Get project from context if not in data
        if 'project' not in validated_data:
            project = self.context.get('project')
            if project:
                validated_data['project'] = project
            else:
                raise serializers.ValidationError({
                    "project": "Project is required"
                })
        
        # Get user from request
        request = self.context.get('request')
        if request and request.user.is_authenticated:
            validated_data['user'] = request.user
        
        # Auto-approve
        validated_data['is_approved'] = True
        
        return super().create(validated_data)
    
    def update(self, instance, validated_data):
        """
        Prevent updating project
        """
        validated_data.pop('project', None)  # Don't allow changing project
        return super().update(instance, validated_data)

class CartItemSerializer(serializers.ModelSerializer):
    project = serializers.SerializerMethodField()  # Changed to SerializerMethodField
    project_id = serializers.PrimaryKeyRelatedField(
        queryset=Project.objects.all(),
        source='project',
        write_only=True
    )
    price = serializers.DecimalField(max_digits=10, decimal_places=2, read_only=True)
    total_price = serializers.DecimalField(max_digits=10, decimal_places=2, read_only=True)

    class Meta:
        model = CartItem
        fields = '__all__'
        read_only_fields = ['created_at']

    def get_project(self, obj):
        # FIXED: Pass context to ProjectSerializer
        serializer = ProjectSerializer(
            obj.project, 
            context=self.context  # Pass context here
        )
        return serializer.data


class CartSerializer(serializers.ModelSerializer):
    items = CartItemSerializer(many=True, read_only=True)
    total_items = serializers.IntegerField(read_only=True)
    total_price = serializers.DecimalField(max_digits=10, decimal_places=2, read_only=True)
    user = UserSerializer(read_only=True)

    class Meta:
        model = Cart
        fields = '__all__'

# class OrderSerializer(serializers.ModelSerializer):
#     user = UserSerializer(read_only=True)
#     project = ProjectSerializer(read_only=True)
#     user_id = serializers.PrimaryKeyRelatedField(
#         queryset=User.objects.all(),
#         source='user',
#         write_only=True
#     )
#     project_id = serializers.PrimaryKeyRelatedField(
#         queryset=Project.objects.all(),
#         source='project',
#         write_only=True
#     )
#     can_download = serializers.SerializerMethodField()

#     class Meta:
#         model = Order
#         fields = '__all__'
#         read_only_fields = ['order_id', 'created_at', 'updated_at', 'completed_at']

#     def get_can_download(self, obj):
#         return obj.payment_status and obj.status == 'completed' and obj.delivered_files
class OrderSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    project = ProjectSerializer(read_only=True)

    user_id = serializers.PrimaryKeyRelatedField(
        queryset=User.objects.all(),
        source='user',
        write_only=True
    )
    project_id = serializers.PrimaryKeyRelatedField(
        queryset=Project.objects.all(),
        source='project',
        write_only=True
    )

    can_download = serializers.SerializerMethodField()
    delivered_file_url = serializers.SerializerMethodField()

    custom_requirements = serializers.SerializerMethodField()
    delivery_message = serializers.SerializerMethodField()
    payment_id = serializers.SerializerMethodField()

    class Meta:
        model = Order
        fields = '__all__'
        read_only_fields = ['order_id', 'created_at', 'updated_at', 'completed_at']

    def get_can_download(self, obj):
        return bool(obj.payment_status and obj.status == 'completed' and obj.delivered_files)

    def get_delivered_file_url(self, obj):
        return obj.delivered_files.url if obj.delivered_files else None

    def _safe_text(self, value):
        if not value:
            return ""
        if isinstance(value, bytes):
            return value.decode('utf-8', 'ignore')
        return str(value)

    def get_custom_requirements(self, obj):
        return self._safe_text(obj.custom_requirements)

    def get_delivery_message(self, obj):
        return self._safe_text(obj.delivery_message)

    def get_payment_id(self, obj):
        return self._safe_text(obj.payment_id)
        
class ReferralSerializer(serializers.ModelSerializer):
    referrer = UserSerializer(read_only=True)
    referred_user = UserSerializer(read_only=True)
    referrer_id = serializers.PrimaryKeyRelatedField(
        queryset=User.objects.all(),
        source='referrer',
        write_only=True
    )
    referred_user_id = serializers.PrimaryKeyRelatedField(
        queryset=User.objects.all(),
        source='referred_user',
        write_only=True
    )

    class Meta:
        model = Referral
        fields = '__all__'
        read_only_fields = ['created_at', 'credited_at']

class ProjectViewSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    project = ProjectSerializer(read_only=True)

    class Meta:
        model = ProjectView
        fields = '__all__'
        read_only_fields = ['viewed_at']

class ProjectPreviewSerializer(serializers.ModelSerializer):
    project = ProjectSerializer(read_only=True)

    class Meta:
        model = ProjectPreview
        fields = '__all__'

class DashboardStatsSerializer(serializers.Serializer):
    total_orders = serializers.IntegerField()
    completed_orders = serializers.IntegerField()
    pending_orders = serializers.IntegerField()
    total_spent = serializers.DecimalField(max_digits=10, decimal_places=2)
    avg_order_value = serializers.DecimalField(max_digits=10, decimal_places=2)
    completion_rate = serializers.FloatField()
    monthly_spending = serializers.DecimalField(max_digits=10, decimal_places=2)
    avg_project_rating = serializers.FloatField()
    recent_activity = serializers.IntegerField()

class AdminDashboardSerializer(serializers.Serializer):
    total_users = serializers.IntegerField()
    new_users_today = serializers.IntegerField()
    new_users_week = serializers.IntegerField()
    total_projects = serializers.IntegerField()
    active_projects = serializers.IntegerField()
    featured_projects = serializers.IntegerField()
    total_orders = serializers.IntegerField()
    completed_orders = serializers.IntegerField()
    pending_orders = serializers.IntegerField()
    total_revenue = serializers.DecimalField(max_digits=10, decimal_places=2)
    today_revenue = serializers.DecimalField(max_digits=10, decimal_places=2)
    week_revenue = serializers.DecimalField(max_digits=10, decimal_places=2)
    
    
#AiService Serializer
# Add these serializers at the end

class UserBehaviorSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    project = ProjectSerializer(read_only=True)
    category = CategorySerializer(read_only=True)

    class Meta:
        model = UserBehavior
        fields = '__all__'

class UserProfileVectorSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)

    class Meta:
        model = UserProfileVector
        fields = '__all__'

class ProjectEmbeddingSerializer(serializers.ModelSerializer):
    project = ProjectSerializer(read_only=True)

    class Meta:
        model = ProjectEmbedding
        fields = '__all__'

class AIChatSessionSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    recommended_projects = ProjectSerializer(many=True, read_only=True)
    messages = serializers.SerializerMethodField()

    class Meta:
        model = AIChatSession
        fields = '__all__'
        read_only_fields = ['session_id', 'created_at', 'updated_at']

    def get_messages(self, obj):
        messages = obj.messages.all()[:50]  # Limit to last 50 messages
        return AIChatMessageSerializer(messages, many=True).data

class AIChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = AIChatMessage
        fields = '__all__'
        read_only_fields = ['created_at']

class RecommendationSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    project = ProjectSerializer(read_only=True)

    class Meta:
        model = Recommendation
        fields = '__all__'

class SearchHistorySerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)

    class Meta:
        model = SearchHistory
        fields = '__all__'
        read_only_fields = ['created_at']

class AnomalyDetectionSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    resolved_by = UserSerializer(read_only=True)

    class Meta:
        model = AnomalyDetection
        fields = '__all__'
        read_only_fields = ['created_at']

class LearningPathSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    projects = serializers.SerializerMethodField()
    progress_percentage = serializers.SerializerMethodField()

    class Meta:
        model = LearningPath
        fields = '__all__'
        read_only_fields = ['created_at', 'updated_at']

    def get_projects(self, obj):
        projects = obj.learningpathproject_set.order_by('order')
        return LearningPathProjectSerializer(projects, many=True).data

    def get_progress_percentage(self, obj):
        return round(obj.progress * 100, 2)

class LearningPathProjectSerializer(serializers.ModelSerializer):
    project = ProjectSerializer(read_only=True)

    class Meta:
        model = LearningPathProject
        fields = '__all__'

class AIRecommendationRequestSerializer(serializers.Serializer):
    limit = serializers.IntegerField(default=10, min_value=1, max_value=50)
    refresh = serializers.BooleanField(default=False)
    categories = serializers.ListField(child=serializers.CharField(), required=False)
    difficulty = serializers.CharField(required=False)

class AISearchRequestSerializer(serializers.Serializer):
    query = serializers.CharField(required=True)
    limit = serializers.IntegerField(default=20, min_value=1, max_value=100)
    category = serializers.CharField(required=False)
    difficulty = serializers.CharField(required=False)
    min_price = serializers.DecimalField(max_digits=10, decimal_places=2, required=False)
    max_price = serializers.DecimalField(max_digits=10, decimal_places=2, required=False)

class AILearningPathRequestSerializer(serializers.Serializer):
    goal = serializers.CharField(required=False)
    duration_hours = serializers.IntegerField(default=50, min_value=10, max_value=200)

class AIChatRequestSerializer(serializers.Serializer):
    message = serializers.CharField(required=True)
    session_id = serializers.CharField(required=False)
    context = serializers.JSONField(required=False)