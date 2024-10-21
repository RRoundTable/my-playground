from google_auth_oauthlib.flow import InstalledAppFlow

# Replace 'client_secrets.json' with the path to your client secrets file
flow = InstalledAppFlow.from_client_secrets_file(
    'client_secret.json',
    scopes=['https://www.googleapis.com/auth/yt-analytics.readonly']
)

credentials = flow.run_local_server(port=0)

# Save the refresh token for future use
refresh_token = credentials.refresh_token
print(f"Refresh Token: {refresh_token}")
