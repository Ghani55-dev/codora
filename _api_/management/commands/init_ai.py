from django.core.management.base import BaseCommand
import os
import sys


def _ensure_django():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'codora.settings')
    try:
        import django
        django.setup()
    except Exception:
        # If django isn't available yet, importing later will raise a clear error
        pass


try:
    from _api_.ai_services import AIService
    from _api_.models import Project, User
except ModuleNotFoundError:
    _ensure_django()
    from _api_.ai_services import AIService
    from _api_.models import Project, User


class Command(BaseCommand):
    help = 'Initialize AI models for all projects and users'

    def handle(self, *args, **options):
        self.stdout.write('Initializing AI models...')

        # Initialize project embeddings
        service = AIService()
        projects = Project.objects.all()
        self.stdout.write(f'Creating embeddings for {projects.count()} projects...')
        for project in projects:
            service.update_project_embedding(project)

        # Initialize user profiles
        users = User.objects.all()
        self.stdout.write(f'Creating profiles for {users.count()} users...')
        for user in users:
            service.update_user_profile(user)

        self.stdout.write(self.style.SUCCESS('AI models initialized successfully!'))


if __name__ == '__main__':
    _ensure_django()
    Command().handle()