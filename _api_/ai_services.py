import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import json
from datetime import datetime, timedelta
from collections import defaultdict
import re
from django.db.models import Count, Avg, Q, Sum, Max
from django.utils import timezone
import random
import joblib
import os
from pathlib import Path

from .models import *

# Update the _extract_features method and related price calculations

class AIService:
    """Main AI service class"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = MinMaxScaler()
        self.model_path = Path(__file__).parent / 'ai_models'
        self.model_path.mkdir(exist_ok=True)
    
    # ===== USER PROFILE UPDATING =====
    
    def update_user_profile(self, user):
        """Update user profile vector based on behavior"""
        from django.db.models import Max
        profile_vector, created = UserProfileVector.objects.get_or_create(user=user)
        
        # Get user behavior data
        behaviors = UserBehavior.objects.filter(user=user)
        orders = Order.objects.filter(user=user, payment_status=True)
        reviews = Review.objects.filter(user=user)
        views = ProjectView.objects.filter(user=user)
        
        # Calculate category preferences
        category_stats = defaultdict(int)
        
        # From purchases
        for order in orders:
            if order.project.category:
                category_stats[order.project.category_id] += 3  # Higher weight for purchases
        
        # From views
        for view in views:
            if view.project.category:
                category_stats[view.project.category_id] += 1
        
        # Normalize category weights
        total = sum(category_stats.values()) or 1
        category_weights = {str(k): v/total for k, v in category_stats.items()}
        
        # Calculate technology preferences
        tech_stats = defaultdict(int)
        for order in orders:
            for tech in order.project.technologies.all():
                tech_stats[tech.id] += 2
        for view in views:
            for tech in view.project.technologies.all():
                tech_stats[tech.id] += 1
        
        # Normalize technology weights
        tech_total = sum(tech_stats.values()) or 1
        tech_weights = {str(k): v/tech_total for k, v in tech_stats.items()}
        
        # Calculate difficulty preference
        difficulty_map = {'beginner': 0, 'intermediate': 0.5, 'advanced': 1}
        difficulty_sum = 0
        difficulty_count = 0
        
        for order in orders:
            difficulty_sum += difficulty_map.get(order.project.difficulty, 0.5)
            difficulty_count += 1
        
        difficulty_preference = difficulty_sum / difficulty_count if difficulty_count > 0 else 0.5
        
        # Calculate price sensitivity using actual price values
        if orders.exists():
            # Get average of final amounts
            avg_price = orders.aggregate(avg=Avg('final_amount'))['avg'] or 0
            
            # Get average of all project base prices (sale_price if available)
            all_projects = Project.objects.filter(is_active=True)
            
            # Calculate average price of all active projects
            total_price = 0
            count = 0
            for project in all_projects:
                # Use sale_price if available, otherwise base_price
                price = project.sale_price if project.sale_price else project.base_price
                total_price += price
                count += 1
            
            all_projects_avg = total_price / count if count > 0 else 0
            
            if all_projects_avg > 0:
                price_sensitivity = min(1.0, avg_price / all_projects_avg)
            else:
                price_sensitivity = 0.5
        else:
            price_sensitivity = 0.5
        
        # Update profile
        profile_vector.category_weights = category_weights
        profile_vector.technology_weights = tech_weights
        profile_vector.difficulty_preference = difficulty_preference
        profile_vector.price_sensitivity = price_sensitivity
        profile_vector.needs_recalculation = False
        
        # Generate embedding
        embedding = self._generate_user_embedding(profile_vector)
        profile_vector.embedding = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        
        profile_vector.save()
        return profile_vector
    
    def _generate_user_embedding(self, profile_vector):
        """Generate embedding vector for user"""
        # Simple embedding based on preferences
        embedding = []
        
        # Add category weights (normalized)
        all_categories = Category.objects.all()
        cat_embedding = [profile_vector.category_weights.get(str(cat.id), 0) 
                        for cat in all_categories]
        embedding.extend(cat_embedding)
        
        # Add technology weights
        all_tech = Technology.objects.all()
        tech_embedding = [profile_vector.technology_weights.get(str(tech.id), 0)
                         for tech in all_tech]
        embedding.extend(tech_embedding)
        
        # Add other preferences
        embedding.append(profile_vector.difficulty_preference)
        embedding.append(profile_vector.price_sensitivity)
        
        # Pad or truncate to fixed size
        target_size = 100
        if len(embedding) < target_size:
            embedding.extend([0] * (target_size - len(embedding)))
        else:
            embedding = embedding[:target_size]
        
        return embedding

    def _text_to_embedding(self, text):
        """Convert text to numeric embedding using TF-IDF vectorizer.

        If the vectorizer hasn't been fitted yet, fit it on all active project texts.
        """
        try:
            # Fit vectorizer on corpus of projects if not yet fitted
            if not getattr(self, '_vectorizer_fitted', False):
                corpus = []
                for p in Project.objects.filter(is_active=True):
                    t = f"{p.title} {p.description or ''} {p.short_description or ''}"
                    t += " " + " ".join(getattr(p, 'features_list', []))
                    t += " " + " ".join(getattr(p, 'learning_outcomes_list', []))
                    corpus.append(t)
                # Always include the provided text so transform shape is stable
                corpus.append(text)
                if corpus:
                    self.vectorizer.fit(corpus)
                    self._vectorizer_fitted = True

            vec = self.vectorizer.transform([text]).toarray()[0]
            return vec.tolist()
        except Exception:
            # Fallback simple bag-of-words frequency vector (limited size)
            words = re.findall(r"\w+", (text or '').lower())
            freq = defaultdict(int)
            for w in words:
                freq[w] += 1
            vals = list(freq.values())[:100]
            if len(vals) < 100:
                vals += [0] * (100 - len(vals))
            return vals
    
    # ===== PROJECT EMBEDDING =====
    
    def update_project_embedding(self, project):
        """Update embedding for a project"""
        from django.db.models import Max
        embedding, created = ProjectEmbedding.objects.get_or_create(project=project)
        
        # Generate text embedding from description
        text = f"{project.title} {project.description} {project.short_description}"
        text += " ".join(project.features_list)
        text += " ".join(project.learning_outcomes_list)
        
        # Simple TF-IDF like embedding (in production, use BERT/SentenceTransformer)
        embedding.description_embedding = self._text_to_embedding(text)
        
        # Generate feature embedding
        features = self._extract_features(project)
        embedding.feature_embedding = features
        
        # Category encoding
        cat_vector = [0] * Category.objects.count()
        if project.category:
            try:
                all_categories = list(Category.objects.all())
                cat_index = all_categories.index(project.category)
                cat_vector[cat_index] = 1
            except ValueError:
                pass  # Category not found in list
        embedding.category_vector = cat_vector
        
        # Technology vector
        tech_vector = [0] * Technology.objects.count()
        try:
            all_tech = list(Technology.objects.all())
            for tech in project.technologies.all():
                tech_index = all_tech.index(tech)
                tech_vector[tech_index] = 1
        except ValueError:
            pass
        embedding.technology_vector = tech_vector
        
        # Calculate popularity score
        total_views = Project.objects.aggregate(total=Sum('views'))['total'] or 1
        total_purchases = Project.objects.aggregate(total=Sum('purchases'))['total'] or 1
        
        view_score = project.views / total_views
        purchase_score = project.purchases / total_purchases
        
        embedding.popularity_score = 0.7 * purchase_score + 0.3 * view_score
        
        # Calculate quality score (based on reviews)
        reviews = Review.objects.filter(project=project, is_approved=True)
        if reviews.exists():
            avg_rating = reviews.aggregate(avg=Avg('rating'))['avg']
            review_count = reviews.count()
            embedding.quality_score = (avg_rating / 5) * min(1, review_count / 10)
        else:
            embedding.quality_score = 0.5
        
        embedding.save()
        
        # Update similar projects
        self._update_similar_projects(project, embedding)
        
        return embedding
    
    def _extract_features(self, project):
        """Extract feature vector from project"""
        features = []
        
        # Difficulty encoding
        difficulty_map = {'beginner': 0, 'intermediate': 0.5, 'advanced': 1}
        features.append(difficulty_map.get(project.difficulty, 0.5))
        
        # Price normalized - use sale_price if available, otherwise base_price
        price = project.sale_price if project.sale_price else project.base_price
        try:
            price_val = float(price) if price is not None else 0.0
        except Exception:
            price_val = 0.0

        # Get max price from all active projects (as float)
        all_projects = Project.objects.filter(is_active=True)
        max_price = 0.0
        for p in all_projects:
            p_price = p.sale_price if p.sale_price else p.base_price
            try:
                p_price_val = float(p_price) if p_price is not None else 0.0
            except Exception:
                p_price_val = 0.0
            if p_price_val > max_price:
                max_price = p_price_val

        max_price = max_price or 1.0  # Avoid division by zero
        features.append(price_val / max_price)
        
        # Feature count
        features.append(len(project.features_list) / 20)  # Normalized
        
        # Text length features
        features.append(len(project.description) / 5000)
        features.append(len(" ".join(project.learning_outcomes_list)) / 1000)
        
        return features
    
    def _update_similar_projects(self, project, embedding):
        """Find and cache similar projects"""
        all_embeddings = ProjectEmbedding.objects.exclude(project=project)
        
        similarities = []
        for other_embedding in all_embeddings:
            if other_embedding.description_embedding:
                # Calculate similarity (cosine similarity)
                sim = self._cosine_similarity(
                    embedding.description_embedding,
                    other_embedding.description_embedding
                )
                similarities.append({
                    'project_id': other_embedding.project.id,
                    'similarity': sim,
                    'title': other_embedding.project.title,
                    'slug': other_embedding.project.slug
                })
        
        # Sort by similarity and take top 10
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        embedding.similar_projects = similarities[:10]
        embedding.save()
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2:
            return 0
        
        # Ensure vectors are same length
        min_len = min(len(vec1), len(vec2))
        vec1 = vec1[:min_len]
        vec2 = vec2[:min_len]
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return dot_product / (norm1 * norm2)
    
    # ===== RECOMMENDATION ENGINE =====
    
    def generate_recommendations(self, user, limit=20):
        """Generate personalized recommendations for user"""
        # Update user profile first
        self.update_user_profile(user)
        
        # Get user profile
        profile = UserProfileVector.objects.filter(user=user).first()
        if not profile:
            profile = self.update_user_profile(user)
        
        # Get all active projects user hasn't purchased
        purchased_ids = Order.objects.filter(
            user=user, 
            payment_status=True
        ).values_list('project_id', flat=True)
        
        candidate_projects = Project.objects.filter(
            is_active=True
        ).exclude(id__in=purchased_ids)
        
        recommendations = []
        
        # Get all projects to calculate max price
        all_active_projects = Project.objects.filter(is_active=True)
        max_price = 0
        for p in all_active_projects:
            price = p.sale_price if p.sale_price else p.base_price
            if price > max_price:
                max_price = price
        max_price = max_price or 1
        
        for project in candidate_projects:
            # Get or create project embedding
            proj_embedding, _ = ProjectEmbedding.objects.get_or_create(project=project)
            
            # Calculate various scores
            content_score = self._calculate_content_score(profile, proj_embedding, max_price)
            collaborative_score = self._calculate_collaborative_score(user, project)
            popularity_score = proj_embedding.popularity_score
            personalization_score = self._calculate_personalization_score(profile, proj_embedding)
            
            # Weighted final score
            final_score = (
                0.4 * content_score +
                0.3 * collaborative_score +
                0.15 * popularity_score +
                0.15 * personalization_score
            )
            
            # Determine recommendation type
            rec_type = self._determine_recommendation_type(
                content_score, collaborative_score, popularity_score
            )
            
            # Generate reason
            reason = self._generate_recommendation_reason(
                project, content_score, collaborative_score, rec_type
            )
            
            recommendations.append({
                'project': project,
                'content_score': content_score,
                'collaborative_score': collaborative_score,
                'popularity_score': popularity_score,
                'personalization_score': personalization_score,
                'final_score': final_score,
                'recommendation_type': rec_type,
                'reason': reason
            })
        
        # Sort by final score
        recommendations.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Save to database
        saved_recommendations = []
        for i, rec in enumerate(recommendations[:limit]):
            recommendation, created = Recommendation.objects.update_or_create(
                user=user,
                project=rec['project'],
                defaults={
                    'content_score': rec['content_score'],
                    'collaborative_score': rec['collaborative_score'],
                    'popularity_score': rec['popularity_score'],
                    'personalization_score': rec['personalization_score'],
                    'final_score': rec['final_score'],
                    'recommendation_type': rec['recommendation_type'],
                    'reason': rec['reason'],
                    'position': i,
                    'is_active': True,
                    'expires_at': timezone.now() + timedelta(days=7)
                }
            )
            saved_recommendations.append(recommendation)
        
        return saved_recommendations
    
    def _calculate_content_score(self, profile, project_embedding, max_price=1):
        """Calculate content-based similarity score"""
        if not profile.embedding or not project_embedding.description_embedding:
            return 0
        
        project = project_embedding.project
        
        # Category match
        category_score = 0
        if project.category_id:
            cat_weight = profile.category_weights.get(str(project.category_id), 0)
            category_score = cat_weight
        
        # Technology match
        tech_score = 0
        project_tech_ids = list(project.technologies.values_list('id', flat=True))
        for tech_id in project_tech_ids:
            tech_score += profile.technology_weights.get(str(tech_id), 0)
        if project_tech_ids:
            tech_score /= len(project_tech_ids)
        
        # Difficulty match
        difficulty_map = {'beginner': 0, 'intermediate': 0.5, 'advanced': 1}
        proj_difficulty = difficulty_map.get(project.difficulty, 0.5)
        difficulty_score = 1 - abs(profile.difficulty_preference - proj_difficulty)
        
        # Price match - use sale_price if available, otherwise base_price
        price = project.sale_price if project.sale_price else project.base_price
        normalized_price = float(price) / max_price
        price_score = 1 - abs(profile.price_sensitivity - normalized_price)
        
        # Combine scores
        content_score = (
            0.3 * category_score +
            0.25 * tech_score +
            0.2 * difficulty_score +
            0.25 * price_score
        )
        
        return content_score
    
    def _calculate_collaborative_score(self, user, project):
        """Calculate collaborative filtering score"""
        # Find similar users who purchased this project
        similar_users = self._find_similar_users(user)
        
        if not similar_users:
            return 0
        
        # Count how many similar users purchased this project
        purchase_count = Order.objects.filter(
            user__in=similar_users,
            project=project,
            payment_status=True
        ).count()
        
        # Normalize by number of similar users
        score = purchase_count / len(similar_users)
        return min(score, 1.0)
    
    def _find_similar_users(self, user, limit=10):
        """Find users with similar behavior"""
        # Get user's purchased categories
        user_categories = Order.objects.filter(
            user=user, 
            payment_status=True
        ).values_list('project__category', flat=True).distinct()
        
        # Find users who purchased same categories
        similar_users = User.objects.filter(
            orders__project__category__in=user_categories,
            orders__payment_status=True
        ).exclude(id=user.id).annotate(
            common_purchases=Count('orders', filter=Q(orders__payment_status=True))
        ).order_by('-common_purchases')[:limit]
        
        return list(similar_users)
    
    def _calculate_personalization_score(self, profile, project_embedding):
        """Calculate personalized score based on user embedding"""
        if not profile.embedding or not project_embedding.description_embedding:
            return 0
        
        # Calculate cosine similarity between user and project embeddings
        user_embedding = np.array(profile.embedding)
        proj_embedding = np.array(project_embedding.description_embedding)
        
        # Ensure same length
        min_len = min(len(user_embedding), len(proj_embedding))
        user_embedding = user_embedding[:min_len]
        proj_embedding = proj_embedding[:min_len]
        
        # Calculate cosine similarity
        dot = np.dot(user_embedding, proj_embedding)
        norm_user = np.linalg.norm(user_embedding)
        norm_proj = np.linalg.norm(proj_embedding)
        
        if norm_user == 0 or norm_proj == 0:
            return 0
        
        similarity = dot / (norm_user * norm_proj)
        return max(0, similarity)  # Ensure non-negative
    
    def _determine_recommendation_type(self, content_score, collaborative_score, popularity_score):
        """Determine the type of recommendation"""
        if content_score > 0.7 and collaborative_score > 0.5:
            return 'personalized'
        elif content_score > collaborative_score and content_score > popularity_score:
            return 'content_based'
        elif collaborative_score > content_score and collaborative_score > popularity_score:
            return 'collaborative'
        elif popularity_score > 0.8:
            return 'trending'
        else:
            return 'complementary'
    
    def _generate_recommendation_reason(self, project, content_score, collaborative_score, rec_type):
        """Generate human-readable reason for recommendation"""
        reasons = {
            'personalized': f"Based on your interests and what similar users liked",
            'content_based': f"Similar to projects you've shown interest in",
            'collaborative': f"Users with similar interests purchased this",
            'trending': f"Currently popular among students",
            'complementary': f"Complements your learning journey",
            'similar_users': f"Users like you enjoyed this project"
        }
        
        base_reason = reasons.get(rec_type, "Recommended for you")
        
        # Add specific details
        details = []
        if content_score > 0.6:
            details.append("matches your interests")
        if collaborative_score > 0.4:
            details.append("highly rated by similar users")
        if project.is_featured:
            details.append("featured project")
        
        if details:
            return f"{base_reason} because it {', '.join(details)}"
        
        return base_reason
    
    # ===== INTELLIGENT SEARCH =====
    
    def intelligent_search(self, query, user=None, filters=None, limit=20):
        """Enhanced search with semantic understanding"""
        # Save search history
        if user and user.is_authenticated:
            SearchHistory.objects.create(
                user=user,
                query=query,
                search_filters=filters or {}
            )
        
        # Detect intent
        intent = self._detect_search_intent(query)
        
        # Base queryset
        projects = Project.objects.filter(is_active=True)
        
        # Apply filters
        if filters:
            if filters.get('category'):
                projects = projects.filter(category__slug=filters['category'])
            if filters.get('difficulty'):
                projects = projects.filter(difficulty=filters['difficulty'])
            if filters.get('min_price'):
                # Handle price filtering without using current_price property
                projects = projects.filter(
                    models.Q(sale_price__gte=filters['min_price']) | 
                    models.Q(base_price__gte=filters['min_price'], sale_price__isnull=True)
                )
            if filters.get('max_price'):
                projects = projects.filter(
                    models.Q(sale_price__lte=filters['max_price']) | 
                    models.Q(base_price__lte=filters['max_price'], sale_price__isnull=True)
                )
        
        # Keyword search
        keyword_results = projects.filter(
            Q(title__icontains=query) |
            Q(description__icontains=query) |
            Q(short_description__icontains=query) |
            Q(technologies__name__icontains=query)
        ).distinct()
        
        # Semantic search (if keyword results are few)
        if keyword_results.count() < 5:
            semantic_results = self._semantic_search(query, projects)
            # Combine results
            combined_ids = list(keyword_results.values_list('id', flat=True))
            combined_ids.extend([p.id for p in semantic_results])
            projects = projects.filter(id__in=combined_ids)
        else:
            projects = keyword_results
        
        # Personalize ranking if user is authenticated
        if user and user.is_authenticated:
            projects = self._personalize_search_ranking(user, projects, query)
        
        # Apply intent-based boosting
        projects = self._apply_intent_boost(projects, intent)
        
        return projects[:limit], intent
    
    def _detect_search_intent(self, query):
        """Detect user intent from search query"""
        query_lower = query.lower()
        
        intent_keywords = {
            'learn': ['learn', 'tutorial', 'guide', 'how to', 'basics'],
            'buy': ['buy', 'purchase', 'order', 'price', 'cost'],
            'compare': ['compare', 'vs', 'difference', 'alternative'],
            'trending': ['popular', 'trending', 'best', 'top', 'hot'],
            'beginner': ['beginner', 'easy', 'simple', 'starter', 'basic'],
            'advanced': ['advanced', 'complex', 'expert', 'professional'],
            'free': ['free', 'cheap', 'low cost', 'budget'],
        }
        
        for intent, keywords in intent_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return intent
        
        return 'general'
    
    def _semantic_search(self, query, projects):
        """Semantic search using embeddings"""
        # In production, use Sentence-BERT or similar
        # This is a simplified version
        
        query_embedding = self._text_to_embedding(query)
        results = []
        
        for project in projects:
            embedding, _ = ProjectEmbedding.objects.get_or_create(project=project)
            if embedding.description_embedding:
                similarity = self._cosine_similarity(
                    query_embedding,
                    embedding.description_embedding
                )
                if similarity > 0.3:  # Threshold
                    results.append((project, similarity))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in results]
    
    def _personalize_search_ranking(self, user, projects, query):
        """Personalize search results ranking"""
        profile = UserProfileVector.objects.filter(user=user).first()
        if not profile:
            return projects
        
        scored_projects = []
        for project in projects:
            embedding, _ = ProjectEmbedding.objects.get_or_create(project=project)
            
            # Calculate personalized score
            personal_score = self._calculate_personalization_score(profile, embedding)
            
            # Combine with relevance
            relevance = 0.7  # Base relevance
            final_score = 0.6 * relevance + 0.4 * personal_score
            
            scored_projects.append((project, final_score))
        
        # Sort by final score
        scored_projects.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored_projects]
    
    def _apply_intent_boost(self, projects, intent):
        """Boost projects based on detected intent"""
        scored_projects = []
        
        for project in projects:
            embedding, _ = ProjectEmbedding.objects.get_or_create(project=project)
            score = embedding.popularity_score
            
            # Intent-based boosting
            if intent == 'beginner' and project.difficulty == 'beginner':
                score *= 1.5
            elif intent == 'advanced' and project.difficulty == 'advanced':
                score *= 1.5
            elif intent == 'free':
                # Check if price is low (using sale_price if available, otherwise base_price)
                price = project.sale_price if project.sale_price else project.base_price
                if price < 1000:
                    score *= 1.3
            elif intent == 'trending':
                score = embedding.popularity_score * 1.4
            elif intent == 'learn':
                # Boost projects with good documentation
                if project.documentation:
                    score *= 1.2
            
            scored_projects.append((project, score))
        
        scored_projects.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in scored_projects]
    
    # ===== LEARNING PATH GENERATION =====
    
    def generate_learning_path(self, user, goal=None):
        """Generate personalized learning path for user"""
        # Get user's current skills
        purchased_projects = Project.objects.filter(
            orders__user=user,
            orders__payment_status=True
        ).distinct()
        
        # Determine starting point
        if purchased_projects.exists():
            # Continue from current level
            avg_difficulty = self._calculate_average_difficulty(purchased_projects)
            target_difficulty = min(1.0, avg_difficulty + 0.2)  # Slightly harder
        else:
            # Start from beginner
            target_difficulty = 0.3
        
        # Select projects for learning path
        difficulty_map = {0.3: 'beginner', 0.6: 'intermediate', 0.9: 'advanced'}
        target_level = min(difficulty_map.keys(), key=lambda x: abs(x - target_difficulty))
        
        # Get suitable projects
        candidate_projects = Project.objects.filter(
            is_active=True,
            difficulty=difficulty_map[target_level]
        ).exclude(id__in=purchased_projects.values_list('id', flat=True))
        
        # Sort by relevance to user interests
        profile = UserProfileVector.objects.filter(user=user).first()
        if profile:
            scored_projects = []
            for project in candidate_projects:
                embedding, _ = ProjectEmbedding.objects.get_or_create(project=project)
                score = self._calculate_personalization_score(profile, embedding)
                scored_projects.append((project, score))
            
            scored_projects.sort(key=lambda x: x[1], reverse=True)
            selected_projects = [p for p, _ in scored_projects[:5]]
        else:
            selected_projects = list(candidate_projects.order_by('-purchases')[:5])
        
        # Create learning path
        if selected_projects:
            learning_path = LearningPath.objects.create(
                user=user,
                title=f"Learning Path for {user.username}",
                description=f"Personalized learning path based on your interests",
                estimated_duration=len(selected_projects) * 10,
                difficulty_progression=[difficulty_map[target_level]] * len(selected_projects),
                generated_by='ai',
                confidence_score=0.8
            )
            
            # Add projects to learning path
            for i, project in enumerate(selected_projects):
                LearningPathProject.objects.create(
                    learning_path=learning_path,
                    project=project,
                    order=i,
                    learning_objectives=[
                        f"Understand {project.title} concepts",
                        "Complete the project implementation",
                        "Apply learned skills to real problems"
                    ],
                    estimated_hours=10
                )
            
            return learning_path
        
        return None
    
    def _calculate_average_difficulty(self, projects):
        """Calculate average difficulty level from 0 to 1"""
        difficulty_values = {
            'beginner': 0.3,
            'intermediate': 0.6,
            'advanced': 0.9
        }
        
        total = 0
        for project in projects:
            total += difficulty_values.get(project.difficulty, 0.5)
        
        return total / len(projects) if projects else 0.3
    
    # ===== ANOMALY DETECTION =====
    
    def detect_anomalies(self):
        """Detect anomalies in user behavior"""
        anomalies = []
        
        # Detect multiple accounts from same IP (simplified)
        recent_users = User.objects.filter(date_joined__gte=timezone.now() - timedelta(hours=1))
        
        # Detect fake reviews
        fake_reviews = self._detect_fake_reviews()
        anomalies.extend(fake_reviews)
        
        # Detect suspicious purchases
        suspicious_purchases = self._detect_suspicious_purchases()
        anomalies.extend(suspicious_purchases)
        
        return anomalies
    
    def _detect_fake_reviews(self):
        """Detect potentially fake reviews"""
        anomalies = []
        
        # Reviews with same content
        duplicate_reviews = Review.objects.values('comment').annotate(
            count=Count('id')
        ).filter(count__gt=3)
        
        for review in duplicate_reviews:
            anomalies.append({
                'type': 'fake_reviews',
                'description': f"Multiple reviews with same content: {review['comment'][:50]}...",
                'confidence': min(0.9, review['count'] / 10)
            })
        
        return anomalies
    
    def _detect_suspicious_purchases(self):
        """Detect suspicious purchase patterns"""
        anomalies = []
        
        # Multiple purchases in very short time
        recent_purchases = Order.objects.filter(
            created_at__gte=timezone.now() - timedelta(minutes=5)
        )
        
        user_purchase_counts = recent_purchases.values('user').annotate(
            count=Count('id')
        ).filter(count__gt=3)
        
        for purchase in user_purchase_counts:
            user = User.objects.get(id=purchase['user'])
            anomalies.append({
                'type': 'suspicious_purchase',
                'description': f"User {user.username} made {purchase['count']} purchases in 5 minutes",
                'confidence': min(0.8, purchase['count'] / 10)
            })
        
        return anomalies

# Create singleton instance
ai_service = AIService()