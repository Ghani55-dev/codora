from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from _api_.views import *
from django.conf import settings
from django.conf.urls.static import static

router = DefaultRouter()
router.register(r'users', UserViewSet)
router.register(r'categories', CategoryViewSet)
router.register(r'technologies', TechnologyViewSet)
router.register(r'projects', ProjectViewSet)
router.register(r'reviews', ReviewViewSet)
router.register(r'orders', OrderViewSet, basename='orders')

# API endpoints matching all your original URLs
urlpatterns = [
    path('admin/', admin.site.urls),
    
    # API routes (matching your original structure)
    path('api/', include(router.urls)),
    
    # ===== ACCOUNTS URLs =====
    path('api/accounts/register/', AuthViewSet.as_view({'post': 'register'}), name='register'),
    path('api/accounts/login/', AuthViewSet.as_view({'post': 'login'}), name='login'),
    path('api/accounts/logout/', AuthViewSet.as_view({'post': 'logout'}), name='logout'),
    path('api/accounts/profile/', UserViewSet.as_view({'get': 'profile', 'put': 'update_profile'}), name='profile'),
    path('api/accounts/dashboard/', DashboardViewSet.as_view({'get': 'stats'}), name='dashboard'),
    path('api/accounts/referrals/', ReferralViewSet.as_view({'get': 'dashboard'}), name='referral_dashboard'),
    
    # Password reset endpoints
    path('api/accounts/password-reset/', PasswordResetViewSet.as_view({'post': 'password_reset'}), name='password_reset'),
    path('api/accounts/password-reset/done/', PasswordResetViewSet.as_view({'get': 'password_reset_done'}), name='password_reset_done'),
    path('api/accounts/password-reset-confirm/<uidb64>/<token>/', PasswordResetViewSet.as_view({'post': 'password_reset_confirm'}), name='password_reset_confirm'),
    path('api/accounts/password-reset-complete/', PasswordResetViewSet.as_view({'get': 'password_reset_complete'}), name='password_reset_complete'),
    
    # ===== CART URLs =====
    path('api/cart/', CartViewSet.as_view({'get': 'list'}), name='cart_view'),
    path('api/cart/add/<int:project_id>/', CartViewSet.as_view({'post': 'add_item'}), name='add_to_cart'),
    path('api/cart/remove/<int:item_id>/', CartViewSet.as_view({'delete': 'remove_item'}), name='remove_from_cart'),
    path('api/cart/update/<int:item_id>/', CartViewSet.as_view({'put': 'update_item'}), name='update_cart_item'),
    path('api/cart/clear/', CartViewSet.as_view({'delete': 'clear'}), name='clear_cart'),
    
    # ===== ORDERS URLs =====
    path('api/orders/', OrderViewSet.as_view({'get': 'list'}), name='order_list'),
    path('api/orders/checkout/', OrderViewSet.as_view({'post': 'checkout'}), name='checkout'),
    path('api/orders/<str:order_id>/', OrderViewSet.as_view({'get': 'retrieve'}), name='order_detail'),
    path('api/orders/<str:order_id>/download/', OrderViewSet.as_view({'get': 'download'}), name='download_order_files'),
    path('api/orders/<str:order_id>/admin-update/', AdminViewSet.as_view({'post': 'update_order'}), name='admin_update_order'),
    
    # ===== PROJECTS URLs =====
    path('api/projects/', ProjectViewSet.as_view({'get': 'list'}), name='project_list'),
    path('api/projects/category/<slug:category_slug>/', ProjectViewSet.as_view({'get': 'list'}), name='project_list_by_category'),
    path('api/projects/<slug:slug>/', ProjectViewSet.as_view({'get': 'retrieve'}), name='project_detail'),
    path('api/projects/<int:project_id>/add-review/', ReviewViewSet.as_view({'post': 'create'}), name='add_review'),
    
    # ===== PROJECT PREVIEW URLs =====
    path('api/preview/<slug:project_slug>/', ProjectPreviewViewSet.as_view({'get': 'preview'}), name='project_preview'),
    path('api/preview/<slug:project_slug>/demo/', ProjectPreviewViewSet.as_view({'get': 'live_demo'}), name='live_demo'),
    
    # ===== ADMIN PANEL URLs =====
    path('api/admin/dashboard/', AdminViewSet.as_view({'get': 'dashboard'}), name='admin_dashboard'),
    path('api/admin/orders/', AdminViewSet.as_view({'get': 'orders'}), name='admin_orders'),
    path('api/admin/users/', AdminViewSet.as_view({'get': 'users'}), name='admin_users'),
    path('api/admin/projects/', AdminViewSet.as_view({'get': 'projects'}), name='admin_projects'),
    
     # AI Endpoints
    path('api/ai/recommendations/', AIRecommendationViewSet.as_view({'get': 'get_recommendations', 'post': 'generate'}), name='ai_recommendations'),
    path('api/ai/recommendations/feedback/', AIRecommendationViewSet.as_view({'post': 'feedback'}), name='ai_recommendations_feedback'),
    
    path('api/ai/search/', AISearchViewSet.as_view({'post': 'intelligent_search'}), name='ai_search'),
    path('api/ai/search/history/', AISearchViewSet.as_view({'get': 'search_history'}), name='ai_search_history'),
    
    path('api/ai/learning-paths/', AILearningPathViewSet.as_view({'post': 'generate_path', 'get': 'my_paths'}), name='ai_learning_paths'),
    path('api/ai/learning-paths/<int:pk>/update-progress/', AILearningPathViewSet.as_view({'post': 'update_progress'}), name='ai_learning_path_progress'),
    
    path('api/ai/chat/', AIChatViewSet.as_view({'post': 'chat'}), name='ai_chat'),
    path('api/ai/chat/sessions/', AIChatViewSet.as_view({'get': 'sessions'}), name='ai_chat_sessions'),
    path('api/ai/chat/end-session/', AIChatViewSet.as_view({'post': 'end_session'}), name='ai_chat_end'),
    
    path('api/ai/analytics/behavior/', AIAnalyticsViewSet.as_view({'get': 'user_behavior'}), name='ai_analytics_behavior'),
    path('api/ai/analytics/anomalies/', AIAnalyticsViewSet.as_view({'get': 'anomalies', 'post': 'resolve_anomaly'}), name='ai_analytics_anomalies'),
    path('api/ai/analytics/recommendations/', AIAnalyticsViewSet.as_view({'get': 'recommendations_analytics'}), name='ai_analytics_recommendations'),
    
    # Home endpoint
    path('api/', home, name='home'),
    
    # Frontend routes (if you still need them)
    #path('', TemplateView.as_view(template_name='index.html'), name='frontend_home'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)