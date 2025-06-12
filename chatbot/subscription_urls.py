from django.urls import path
from . import subscription_views

urlpatterns = [
    path('subscription/', subscription_views.subscription_page, name='subscription_page'),
    path('subscription/needed/', subscription_views.subscription_needed, name='subscription_needed'),
    path('subscription/process-payment/', subscription_views.process_payment, name='process_payment'),
    path('subscription/history/', subscription_views.payment_history, name='payment_history'),
    path('subscription/cancel/', subscription_views.cancel_subscription, name='cancel_subscription'),
]
