from django.apps import AppConfig

class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = '_api_'  # This must match your directory name
    
    def ready(self):
        # import _api_.signals  # Import signals when app is ready
        pass
