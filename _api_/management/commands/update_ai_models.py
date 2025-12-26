from django.core.management.base import BaseCommand
from _api_.ai_services import AIService
from _api_.models import Project, User

class Command(BaseCommand):
    help = 'Update all AI models and embeddings'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--models',
            type=str,
            help='Models to update: all, projects, users, recommendations'
        )
    
    def handle(self, *args, **options):
        models_to_update = options.get('models', 'all')
        
        if models_to_update in ['all', 'projects']:
            self.stdout.write('Updating project embeddings...')
            projects = Project.objects.all()
            for project in projects:
                AIService.update_project_embedding(project)
            self.stdout.write(f'Updated {projects.count()} project embeddings')
        
        if models_to_update in ['all', 'users']:
            self.stdout.write('Updating user profiles...')
            users = User.objects.all()
            for user in users:
                AIService.update_user_profile(user)
            self.stdout.write(f'Updated {users.count()} user profiles')
        
        if models_to_update in ['all', 'recommendations']:
            self.stdout.write('Generating recommendations...')
            users = User.objects.all()
            for user in users:
                AIService.generate_recommendations(user, limit=20)
            self.stdout.write(f'Generated recommendations for {users.count()} users')
        
        self.stdout.write(self.style.SUCCESS('AI models updated successfully!'))