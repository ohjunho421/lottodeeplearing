#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

def delete_pycache(base_dir):
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name == "__pycache__":
                pycache_path = os.path.join(root, dir_name)
                for pycache_file in os.listdir(pycache_path):
                    os.remove(os.path.join(pycache_path, pycache_file))
                os.rmdir(pycache_path)

delete_pycache(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Lottobot.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
