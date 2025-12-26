from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
from django.contrib.auth import get_user_model
from .models import Cart  # Make sure this import is correct
from .models import *
from .ai_services import ai_service

User = get_user_model()

@receiver(post_save, sender=User)
def create_user_cart(sender, instance, created, **kwargs):
    """Create a cart for new users"""
    if created:
        Cart.objects.create(user=instance)

@receiver(post_save, sender=User)
def save_user_cart(sender, instance, **kwargs):
    """Ensure user has a cart"""
    if not hasattr(instance, 'cart'):
        Cart.objects.create(user=instance)
        
        
        

@receiver(post_save, sender=Order)
def update_ai_on_order(sender, instance, created, **kwargs):
    """Update AI models when order is created/updated"""
    if created and instance.payment_status:
        # Update project embedding (popularity changes)
        ai_service.update_project_embedding(instance.project)
        
        # Update user profile
        ai_service.update_user_profile(instance.user)
        
        # Log behavior
        UserBehavior.objects.create(
            user=instance.user,
            action='purchase',
            project=instance.project,
            duration=0
        )

@receiver(post_save, sender=ProjectView)
def update_ai_on_view(sender, instance, created, **kwargs):
    """Update AI models when project is viewed"""
    if created:
        # Update project embedding (view count changes)
        ai_service.update_project_embedding(instance.project)
        
        # Update user profile
        ai_service.update_user_profile(instance.user)
        
        # Log behavior
        UserBehavior.objects.create(
            user=instance.user,
            action='view',
            project=instance.project,
            duration=30  # Assume 30 seconds average view time
        )

@receiver(post_save, sender=Review)
def update_ai_on_review(sender, instance, created, **kwargs):
    """Update AI models when review is added"""
    if created and instance.is_approved:
        # Update project embedding (quality score changes)
        ai_service.update_project_embedding(instance.project)
        
        # Log behavior
        UserBehavior.objects.create(
            user=instance.user,
            action='review',
            project=instance.project,
            rating=instance.rating
        )

@receiver(post_save, sender=User)
def create_user_ai_profile(sender, instance, created, **kwargs):
    """Create AI profile for new users"""
    if created:
        UserProfileVector.objects.get_or_create(user=instance)

@receiver(post_save, sender=Project)
def create_project_embedding(sender, instance, created, **kwargs):
    """Create embedding for new projects"""
    if created:
        ai_service.update_project_embedding(instance)