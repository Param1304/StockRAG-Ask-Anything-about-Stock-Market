from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(f"Streamlit app is live at {public_url}")
