import babyagi
import os


app = babyagi.create_app('/dashboard')

# Add API keys to functionz key store
babyagi.add_key_wrapper('deepseek_api_key', os.environ['DEEPSEEK_API_KEY'])
babyagi.add_key_wrapper('OPENAI_API_KEY', os.environ['OPENAI_API_KEY'])


@app.route('/')
def home():
    return f"Welcome to the main app. Visit <a href=\"/dashboard\">/dashboard</a> for BabyAGI dashboard."

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
