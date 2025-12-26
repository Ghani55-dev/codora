from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils import timezone
from decimal import Decimal
import uuid

class User(AbstractUser):
    phone = models.CharField(max_length=15, blank=True)
    college = models.CharField(max_length=200, blank=True)
    year = models.CharField(max_length=50, blank=True)
    course = models.CharField(max_length=200, blank=True)
    wallet_balance = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    profile_image = models.ImageField(upload_to='profile_pics/', blank=True, null=True)
    email_verified = models.BooleanField(default=False)
    referral_code = models.CharField(max_length=20, unique=True, blank=True)
    referred_by = models.ForeignKey('self', on_delete=models.SET_NULL, null=True, blank=True)
    total_referrals = models.PositiveIntegerField(default=0)
    referral_earnings = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    
    # Fix the reverse accessor conflicts
    groups = models.ManyToManyField(
        'auth.Group',
        verbose_name='groups',
        blank=True,
        help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.',
        related_name="api_user_set",
        related_query_name="api_user",
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        verbose_name='user permissions',
        blank=True,
        help_text='Specific permissions for this user.',
        related_name="api_user_set",
        related_query_name="api_user",
    )

    def add_referral_bonus(self, amount=100):
        self.wallet_balance += amount
        self.referral_earnings += amount
        self.total_referrals += 1
        self.save(update_fields=['wallet_balance', 'referral_earnings', 'total_referrals'])

    def __str__(self):
        return f"{self.username} ({self.college})"

class Category(models.Model):
    name = models.CharField(max_length=100)
    slug = models.SlugField(unique=True)
    description = models.TextField(blank=True)
    icon = models.CharField(max_length=50, default='fas fa-code')

    def __str__(self):
        return self.name

class Technology(models.Model):
    name = models.CharField(max_length=100)
    icon = models.CharField(max_length=50, blank=True)

    def __str__(self):
        return self.name

class Project(models.Model):
    DIFFICULTY_CHOICES = [
        ('beginner', 'Beginner'),
        ('intermediate', 'Intermediate'),
        ('advanced', 'Advanced'),
    ]
    
    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    description = models.TextField()
    short_description = models.CharField(max_length=300)
    base_price = models.DecimalField(max_digits=10, decimal_places=2)
    sale_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='projects')
    technologies = models.ManyToManyField(Technology)
    difficulty = models.CharField(max_length=20, choices=DIFFICULTY_CHOICES)
    features = models.TextField(help_text="List features separated by |")
    learning_outcomes = models.TextField(help_text="What student will learn")
    source_code = models.FileField(upload_to='projects/source/', blank=True)
    documentation = models.FileField(upload_to='projects/docs/', blank=True)
    report = models.FileField(upload_to='projects/reports/', blank=True)
    presentation = models.FileField(upload_to='projects/ppt/', blank=True)
    thumbnail = models.ImageField(upload_to='projects/thumbnails/')
    screenshots = models.ImageField(upload_to='projects/screenshots/', blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    is_featured = models.BooleanField(default=False)
    views = models.PositiveIntegerField(default=0)
    purchases = models.PositiveIntegerField(default=0)
    current_price_db = models.DecimalField(
        max_digits=10, 
        decimal_places=2, 
        null=True, 
        blank=True,
        help_text="Auto-calculated: sale_price if available, otherwise base_price"
    )
    def save(self, *args, **kwargs):
        # Auto-calculate current_price_db before saving
        self.current_price_db = self.sale_price if self.sale_price else self.base_price
        super().save(*args, **kwargs)
    
    @property
    def current_price(self):
        # Keep the property for backward compatibility
        if hasattr(self, 'current_price_db') and self.current_price_db:
            return self.current_price_db
        return self.sale_price if self.sale_price else self.base_price

    # @property
    # def current_price(self):
    #     return self.sale_price if self.sale_price else self.base_price

    @property
    def features_list(self):
        return self.features.split('|') if self.features else []

    @property
    def learning_outcomes_list(self):
        return self.learning_outcomes.split('|') if self.learning_outcomes else []

    def __str__(self):
        return self.title

class Review(models.Model):
    RATING_CHOICES = [
        (1, '★☆☆☆☆'),
        (2, '★★☆☆☆'),
        (3, '★★★☆☆'),
        (4, '★★★★☆'),
        (5, '★★★★★'),
    ]
    
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='reviews')
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    rating = models.IntegerField(choices=RATING_CHOICES, default=5)
    comment = models.TextField()
    is_approved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ['project', 'user']
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - {self.project.title} ({self.rating}/5)"

class ProjectView(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    viewed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-viewed_at']
        unique_together = ['user', 'project']
        verbose_name = 'Project View'
        verbose_name_plural = 'Project Views'

    def __str__(self):
        return f"{self.user.username} viewed {self.project.title}"

class Cart(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='cart')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    @property
    def total_items(self):
        return self.items.count()

    @property
    def total_price(self):
        return sum(item.total_price for item in self.items.all())

    def __str__(self):
        return f"Cart of {self.user.username}"

class CartItem(models.Model):
    cart = models.ForeignKey(Cart, on_delete=models.CASCADE, related_name='items')
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    custom_requirements = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    @property
    def price(self):
        return self.project.current_price

    @property
    def total_price(self):
        return self.project.current_price

    def __str__(self):
        return f"{self.project.title} in cart"

# class Order(models.Model):
#     STATUS_CHOICES = [
#         ('pending', 'Pending'),
#         ('processing', 'Processing'),
#         ('completed', 'Completed'),
#         ('cancelled', 'Cancelled'),
#     ]
    
#     order_id = models.CharField(max_length=20, unique=True)
#     user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='orders')
#     project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='orders')
#     amount = models.DecimalField(max_digits=10, decimal_places=2)
#     discount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
#     final_amount = models.DecimalField(max_digits=10, decimal_places=2)
#     custom_requirements = models.TextField(blank=True)
#     requirements_file = models.FileField(upload_to='orders/requirements/', blank=True)
#     status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
#     payment_status = models.BooleanField(default=False)
#     payment_id = models.CharField(max_length=100, blank=True)
#     created_at = models.DateTimeField(auto_now_add=True)
#     updated_at = models.DateTimeField(auto_now=True)
#     completed_at = models.DateTimeField(null=True, blank=True)
#     delivered_files = models.FileField(upload_to='orders/delivered/', blank=True)
#     delivery_message = models.TextField(blank=True)
#     download_count = models.PositiveIntegerField(default=0)
#     last_download_at = models.DateTimeField(null=True, blank=True)

#     def save(self, *args, **kwargs):
#         if not self.pk:
#             if not self.order_id:
#                 last_order = Order.objects.order_by('-id').first()
#                 if last_order:
#                     last_num = int(last_order.order_id.split('-')[1])
#                     new_num = last_num + 1
#                 else:
#                     new_num = 1
#                 self.order_id = f"COD-{new_num:05d}"
#             if self.delivered_files:
#                 self.download_count = 0
#         super().save(*args, **kwargs)

#     def __str__(self):
#         return f"Order {self.order_id} - {self.user.username}"
class Order(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
    ]
    
    order_id = models.CharField(max_length=20, unique=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='orders')
    project = models.ForeignKey(Project, on_delete=models.CASCADE, related_name='orders')
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    discount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    final_amount = models.DecimalField(max_digits=10, decimal_places=2)
    custom_requirements = models.TextField(blank=True)
    requirements_file = models.FileField(upload_to='orders/requirements/', blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    payment_status = models.BooleanField(default=False)
    payment_id = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    delivered_files = models.FileField(upload_to='orders/delivered/', blank=True)
    delivery_message = models.TextField(blank=True)
    download_count = models.PositiveIntegerField(default=0)
    last_download_at = models.DateTimeField(null=True, blank=True)

    def save(self, *args, **kwargs):
        # Clean ALL text fields before saving
        self._clean_all_text_fields()
        
        if not self.pk:
            if not self.order_id:
                last_order = Order.objects.order_by('-id').first()
                if last_order:
                    last_num = int(last_order.order_id.split('-')[1])
                    new_num = last_num + 1
                else:
                    new_num = 1
                self.order_id = f"COD-{new_num:05d}"
            if self.delivered_files:
                self.download_count = 0
        super().save(*args, **kwargs)

    def _clean_all_text_fields(self):
        """Clean all text fields to prevent encoding issues"""
        # List all text/char fields
        text_fields_to_clean = [
            'delivery_message',
            'custom_requirements', 
            'payment_id',
            'status'  # status is also a CharField
        ]
        
        for field_name in text_fields_to_clean:
            value = getattr(self, field_name)
            if value:
                cleaned_value = self._clean_string(value)
                setattr(self, field_name, cleaned_value)

    def _clean_string(self, value):
        """Comprehensive string cleaning for various encodings"""
        if value is None:
            return ""
        
        if isinstance(value, str):
            # Already a string, clean it
            return self._clean_utf8_string(value)
        elif isinstance(value, bytes):
            # Try multiple encodings
            return self._decode_bytes(value)
        else:
            # Convert to string and clean
            return self._clean_utf8_string(str(value))

    def _clean_utf8_string(self, text):
        """Clean a UTF-8 string by removing/replacing invalid characters"""
        try:
            # Try to encode as UTF-8 first
            text.encode('utf-8')
            return text
        except UnicodeEncodeError:
            # Replace problematic characters
            import re
            # Remove non-ASCII characters
            text = re.sub(r'[^\x00-\x7F]+', '', text)
            return text

    def _decode_bytes(self, byte_data):
        """Try multiple encodings to decode bytes"""
        encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'windows-1252']
        
        for encoding in encodings_to_try:
            try:
                decoded = byte_data.decode(encoding)
                # Now clean the decoded string
                return self._clean_utf8_string(decoded)
            except (UnicodeDecodeError, UnicodeEncodeError):
                continue
        
        # If all encodings fail, return empty string
        return ""

    def __str__(self):
        return f"Order {self.order_id} - {self.user.username}"
class Referral(models.Model):
    referrer = models.ForeignKey(User, on_delete=models.CASCADE, related_name='referrals_made')
    referred_user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='referral_received')
    bonus_amount = models.DecimalField(max_digits=10, decimal_places=2, default=100)
    credited = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    credited_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        unique_together = ['referrer', 'referred_user']

    def __str__(self):
        return f"{self.referrer.username} → {self.referred_user.username}"

    def credit_bonus(self):
        if not self.credited:
            self.referrer.add_referral_bonus(self.bonus_amount)
            self.credited = True
            self.credited_at = timezone.now()
            self.save()
            return True
        return False

class ProjectPreview(models.Model):
    project = models.OneToOneField(Project, on_delete=models.CASCADE, related_name='preview')
    preview_file = models.FileField(upload_to='project_previews/', blank=True)
    live_demo_url = models.URLField(blank=True)
    description = models.TextField(blank=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"Preview for {self.project.title}"
    
    
    
#AIclass AIRequest(models.Model):
class UserBehavior(models.Model):
    """Tracks user behavior patterns"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='behaviors')
    action = models.CharField(max_length=50)  # view, purchase, search, review, etc.
    project = models.ForeignKey(Project, on_delete=models.SET_NULL, null=True, blank=True)
    category = models.ForeignKey(Category, on_delete=models.SET_NULL, null=True, blank=True)
    search_query = models.TextField(blank=True)
    rating = models.IntegerField(null=True, blank=True)
    duration = models.IntegerField(default=0)  # Time spent in seconds
    device_info = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'action']),
            models.Index(fields=['created_at']),
        ]
    
    def __str__(self):
        return f"{self.user.username} - {self.action}"

class UserProfileVector(models.Model):
    """Stores AI-generated user profile vectors for recommendations"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile_vector')
    
    # Category preferences (normalized 0-1)
    category_weights = models.JSONField(default=dict)
    
    # Technology preferences
    technology_weights = models.JSONField(default=dict)
    
    # Difficulty preference (beginner: 0, intermediate: 0.5, advanced: 1)
    difficulty_preference = models.FloatField(default=0.5)
    
    # Price sensitivity (0: budget, 1: premium)
    price_sensitivity = models.FloatField(default=0.5)
    
    # Project feature preferences
    feature_weights = models.JSONField(default=dict)
    
    # Learning style preferences
    learning_style = models.JSONField(default=dict)
    
    # Vector embedding (for ML models)
    embedding = models.JSONField(default=list, blank=True)
    
    last_updated = models.DateTimeField(auto_now=True)
    needs_recalculation = models.BooleanField(default=True)
    
    def __str__(self):
        return f"Profile vector for {self.user.username}"

class ProjectEmbedding(models.Model):
    """Stores AI-generated project embeddings for similarity search"""
    project = models.OneToOneField(Project, on_delete=models.CASCADE, related_name='embedding')
    
    # Text embedding from project description
    description_embedding = models.JSONField(default=list)
    
    # Feature embedding
    feature_embedding = models.JSONField(default=list)
    
    # Category encoding
    category_vector = models.JSONField(default=list)
    
    # Technology vector
    technology_vector = models.JSONField(default=list)
    
    # Popularity score (normalized)
    popularity_score = models.FloatField(default=0.0)
    
    # Quality score
    quality_score = models.FloatField(default=0.0)
    
    # Similar projects (cached)
    similar_projects = models.JSONField(default=list, blank=True)
    
    last_updated = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Embedding for {self.project.title}"

class AIChatSession(models.Model):
    """AI chatbot sessions for user assistance"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='ai_sessions')
    session_id = models.CharField(max_length=100, unique=True)
    
    # Chat context
    context = models.JSONField(default=dict)
    
    # Project recommendations from chat
    recommended_projects = models.ManyToManyField(Project, blank=True)
    
    # User intent detected
    detected_intent = models.CharField(max_length=100, blank=True)
    
    # Chat summary
    summary = models.TextField(blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"AI Session {self.session_id} - {self.user.username}"

class AIChatMessage(models.Model):
    """Individual messages in AI chat sessions"""
    MESSAGE_TYPES = [
        ('user', 'User Message'),
        ('ai', 'AI Response'),
        ('system', 'System Message'),
    ]
    
    session = models.ForeignKey(AIChatSession, on_delete=models.CASCADE, related_name='messages')
    message_type = models.CharField(max_length=10, choices=MESSAGE_TYPES)
    content = models.TextField()
    metadata = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['created_at']
    
    def __str__(self):
        return f"{self.message_type}: {self.content[:50]}..."

class Recommendation(models.Model):
    """Stores AI-generated recommendations"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='recommendations')
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    
    # Recommendation scores
    content_score = models.FloatField(default=0.0)  # Based on content similarity
    collaborative_score = models.FloatField(default=0.0)  # Based on similar users
    popularity_score = models.FloatField(default=0.0)  # Based on trending
    personalization_score = models.FloatField(default=0.0)  # Based on user profile
    
    # Final score
    final_score = models.FloatField(default=0.0)
    
    # Reason for recommendation
    reason = models.TextField(blank=True)
    
    # Recommendation type
    RECOMMENDATION_TYPES = [
        ('content_based', 'Content Based'),
        ('collaborative', 'Collaborative Filtering'),
        ('trending', 'Trending'),
        ('similar_users', 'Similar Users'),
        ('complementary', 'Complementary'),
        ('personalized', 'Personalized'),
    ]
    recommendation_type = models.CharField(max_length=20, choices=RECOMMENDATION_TYPES)
    
    # Display properties
    position = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-final_score', 'position']
        unique_together = ['user', 'project']
    
    def __str__(self):
        return f"Recommendation: {self.user.username} -> {self.project.title}"

class SearchHistory(models.Model):
    """Tracks user searches for improving search relevance"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='searches')
    query = models.CharField(max_length=255)
    
    # Search results metadata
    results_count = models.IntegerField(default=0)
    clicked_results = models.JSONField(default=list, blank=True)
    search_filters = models.JSONField(default=dict, blank=True)
    
    # Semantic analysis
    detected_intent = models.CharField(max_length=100, blank=True)
    query_embedding = models.JSONField(default=list, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'created_at']),
            models.Index(fields=['query']),
        ]
    
    def __str__(self):
        return f"{self.user.username}: {self.query[:50]}"

class AnomalyDetection(models.Model):
    """Tracks anomalies and suspicious activities"""
    ANOMALY_TYPES = [
        ('multiple_accounts', 'Multiple Accounts'),
        ('fake_reviews', 'Fake Reviews'),
        ('suspicious_purchase', 'Suspicious Purchase'),
        ('abuse_pattern', 'Abuse Pattern'),
        ('spam_activity', 'Spam Activity'),
        ('fraudulent_payment', 'Fraudulent Payment'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    anomaly_type = models.CharField(max_length=50, choices=ANOMALY_TYPES)
    description = models.TextField()
    confidence_score = models.FloatField(default=0.0)
    metadata = models.JSONField(default=dict)
    
    # Action taken
    is_resolved = models.BooleanField(default=False)
    action_taken = models.CharField(max_length=100, blank=True)
    resolved_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='resolved_anomalies')
    resolved_at = models.DateTimeField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-confidence_score', '-created_at']
    
    def __str__(self):
        return f"{self.anomaly_type} - {self.confidence_score:.2f}"

class LearningPath(models.Model):
    """AI-generated learning paths for users"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='learning_paths')
    title = models.CharField(max_length=200)
    description = models.TextField(blank=True)
    
    # Path details
    projects = models.ManyToManyField(Project, through='LearningPathProject')
    estimated_duration = models.IntegerField(default=0)  # in hours
    difficulty_progression = models.JSONField(default=list)
    
    # AI metadata
    generated_by = models.CharField(max_length=50, default='ai')
    confidence_score = models.FloatField(default=0.0)
    
    is_active = models.BooleanField(default=True)
    completed = models.BooleanField(default=False)
    progress = models.FloatField(default=0.0)  # 0 to 1
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.title} - {self.user.username}"

class LearningPathProject(models.Model):
    """Projects in learning paths with order and metadata"""
    learning_path = models.ForeignKey(LearningPath, on_delete=models.CASCADE)
    project = models.ForeignKey(Project, on_delete=models.CASCADE)
    order = models.IntegerField(default=0)
    
    # Learning objectives for this step
    learning_objectives = models.JSONField(default=list)
    
    # Prerequisites
    prerequisites = models.JSONField(default=list)
    
    # Estimated time
    estimated_hours = models.IntegerField(default=10)
    
    # Completion tracking
    is_completed = models.BooleanField(default=False)
    completed_at = models.DateTimeField(null=True, blank=True)
    user_rating = models.IntegerField(null=True, blank=True)
    
    class Meta:
        ordering = ['order']
        unique_together = ['learning_path', 'project']
    
    def __str__(self):
        return f"{self.learning_path.title} - Step {self.order}: {self.project.title}"