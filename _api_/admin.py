from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import Group
from .models import *

# Add these to your existing admin.py
# Unregister default Group if you don't need it
admin.site.unregister(Group)

class CustomUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'college', 'course', 'wallet_balance', 'is_staff')
    list_filter = ('is_staff', 'is_superuser', 'college')
    fieldsets = UserAdmin.fieldsets + (
        ('Student Info', {'fields': ('phone', 'college', 'year', 'course', 'profile_image')}),
        ('Wallet & Referrals', {'fields': ('wallet_balance', 'referral_code', 'referred_by', 'total_referrals', 'referral_earnings')}),
    )
    add_fieldsets = UserAdmin.add_fieldsets + (
        ('Student Info', {'fields': ('phone', 'college', 'year', 'course')}),
    )

# Register your User model
admin.site.register(User, CustomUserAdmin)

# Other admin registrations remain the same
@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ('name', 'slug', 'icon')
    prepopulated_fields = {'slug': ('name',)}
    list_per_page = 20

@admin.register(Technology)
class TechnologyAdmin(admin.ModelAdmin):
    list_display = ('name', 'icon')
    list_per_page = 20

@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ('title', 'category', 'base_price', 'current_price', 'difficulty', 'is_active', 'is_featured', 'views', 'purchases')
    list_filter = ('category', 'difficulty', 'is_active', 'is_featured')
    prepopulated_fields = {'slug': ('title',)}
    filter_horizontal = ('technologies',)
    search_fields = ('title', 'description')
    list_per_page = 20
    readonly_fields = ('views', 'purchases')

@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = ('project', 'user', 'rating', 'is_approved', 'created_at')
    list_filter = ('rating', 'is_approved')
    search_fields = ('project__title', 'user__username', 'comment')
    list_per_page = 20
    readonly_fields = ('created_at', 'updated_at')

@admin.register(Cart)
class CartAdmin(admin.ModelAdmin):
    list_display = ('user', 'total_items', 'total_price', 'created_at')
    readonly_fields = ('total_items', 'total_price', 'created_at', 'updated_at')
    list_per_page = 20

@admin.register(CartItem)
class CartItemAdmin(admin.ModelAdmin):
    list_display = ('cart', 'project', 'created_at')
    list_filter = ('cart__user', 'created_at')
    list_per_page = 20
    readonly_fields = ('created_at',)

@admin.register(Order)
class OrderAdmin(admin.ModelAdmin):
    list_display = ('order_id', 'user', 'project', 'status', 'payment_status', 'final_amount', 'created_at')
    list_filter = ('status', 'payment_status', 'created_at')
    search_fields = ('order_id', 'user__username', 'project__title')
    readonly_fields = ('order_id', 'created_at', 'updated_at')
    list_per_page = 20
    
    def get_readonly_fields(self, request, obj=None):
        if obj:  # editing an existing object
            return self.readonly_fields + ('user', 'project', 'amount', 'final_amount')
        return self.readonly_fields

@admin.register(Referral)
class ReferralAdmin(admin.ModelAdmin):
    list_display = ('referrer', 'referred_user', 'bonus_amount', 'credited', 'created_at')
    list_filter = ('credited', 'created_at')
    list_per_page = 20
    readonly_fields = ('created_at', 'credited_at')

@admin.register(ProjectView)
class ProjectViewAdmin(admin.ModelAdmin):
    list_display = ('user', 'project', 'viewed_at')
    list_filter = ('viewed_at',)
    list_per_page = 20
    readonly_fields = ('viewed_at',)

@admin.register(ProjectPreview)
class ProjectPreviewAdmin(admin.ModelAdmin):
    list_display = ('project', 'is_active')
    list_filter = ('is_active',)
    list_per_page = 20

# Customize admin site header
admin.site.site_header = "Codora Administration"
admin.site.site_title = "Codora Admin"
admin.site.index_title = "Welcome to Codora Admin"


#AI Models Admin Registration

@admin.register(UserBehavior)
class UserBehaviorAdmin(admin.ModelAdmin):
    list_display = ('user', 'action', 'project', 'created_at')
    list_filter = ('action', 'created_at')
    search_fields = ('user__username', 'project__title')

@admin.register(UserProfileVector)
class UserProfileVectorAdmin(admin.ModelAdmin):
    list_display = ('user', 'difficulty_preference', 'price_sensitivity', 'last_updated')
    readonly_fields = ('embedding', 'last_updated')

@admin.register(ProjectEmbedding)
class ProjectEmbeddingAdmin(admin.ModelAdmin):
    list_display = ('project', 'popularity_score', 'quality_score', 'last_updated')
    readonly_fields = ('similar_projects', 'last_updated')

@admin.register(AIChatSession)
class AIChatSessionAdmin(admin.ModelAdmin):
    list_display = ('session_id', 'user', 'detected_intent', 'created_at', 'is_active')
    list_filter = ('is_active', 'detected_intent')
    readonly_fields = ('session_id', 'created_at', 'updated_at')

@admin.register(AIChatMessage)
class AIChatMessageAdmin(admin.ModelAdmin):
    list_display = ('session', 'message_type', 'content_preview', 'created_at')
    list_filter = ('message_type',)
    readonly_fields = ('created_at',)
    
    def content_preview(self, obj):
        return obj.content[:50] + '...' if len(obj.content) > 50 else obj.content

@admin.register(Recommendation)
class RecommendationAdmin(admin.ModelAdmin):
    list_display = ('user', 'project', 'recommendation_type', 'final_score', 'is_active')
    list_filter = ('recommendation_type', 'is_active')
    search_fields = ('user__username', 'project__title')

@admin.register(SearchHistory)
class SearchHistoryAdmin(admin.ModelAdmin):
    list_display = ('user', 'query_preview', 'results_count', 'created_at')
    list_filter = ('created_at',)
    readonly_fields = ('created_at',)
    
    def query_preview(self, obj):
        return obj.query[:50] + '...' if len(obj.query) > 50 else obj.query

@admin.register(AnomalyDetection)
class AnomalyDetectionAdmin(admin.ModelAdmin):
    list_display = ('anomaly_type', 'description_preview', 'confidence_score', 'is_resolved', 'created_at')
    list_filter = ('anomaly_type', 'is_resolved')
    readonly_fields = ('created_at', 'resolved_at')
    
    def description_preview(self, obj):
        return obj.description[:50] + '...' if len(obj.description) > 50 else obj.description

@admin.register(LearningPath)
class LearningPathAdmin(admin.ModelAdmin):
    list_display = ('user', 'title', 'progress_percentage', 'is_active', 'created_at')
    list_filter = ('is_active', 'generated_by')
    readonly_fields = ('created_at', 'updated_at')
    
    def progress_percentage(self, obj):
        return f"{obj.progress * 100:.1f}%"

@admin.register(LearningPathProject)
class LearningPathProjectAdmin(admin.ModelAdmin):
    list_display = ('learning_path', 'project', 'order', 'is_completed')
    list_filter = ('is_completed',)
    ordering = ('learning_path', 'order')