from huggingface_hub import HfApi

# --- Configuration ---
TOKEN = ""  # Place your token here (starts with hf_)
REPO_ID = "Keyvan1986/Emma-Classification_Model"  # Exact repository ID of the model created
# Note: Added raw string prefix (r) before the path
LOCAL_FOLDER = r"C:\Users\keyva\MMPL_gpt\Classification Model\final_xlm_r_model_router"

# Windows path example: "C:\\Users\\Keyvan\\Project\\final_xlm_r_model_router"
# Or if the folder is in the current working directory: "./final_xlm_r_model_router"

api = HfApi()

print("🚀 Starting upload to Hugging Face...")

# Uploads all contents of the local folder to the root of the repository
api.upload_folder(
    folder_path=LOCAL_FOLDER,
    repo_id=REPO_ID,
    repo_type="model",
    token=TOKEN
)

print("✅ Upload completed successfully!")
