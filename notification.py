from pyfcm import FCMNotification

# Initialize FCM with your server key
server_key = "AAAAPUAdAtU:APA91bE2B-NDfkt1clUjnk-02APn8z9J_lcuvrpDNoF1lnbfxwSkxauhsrHxVQWXnB0G3m_jGE-ylmEqfVvtfEtWwLmX14DyuIxmfF4jqUtzIQ14eMZ2iMVP2Jx6nsvXYy82_vTdVJjk"
push_service = FCMNotification(api_key=server_key)

# Define the notification message and data payload
# message_title = "Test Notification"
# message_body = "Criminal found"
# data_payload = {
#     "message": "Criminal found."
# }

# The topic to which all devices are subscribed
topic = "news_updates"

# Send the notification with data payload to the specified topic

def notification(message_title,message_body,data_payload):
    result = push_service.notify_topic_subscribers(
        topic_name=topic,
        message_title=message_title,
        message_body=message_body,
        data_message=data_payload  # Add data payload here
    )

    print(result)
