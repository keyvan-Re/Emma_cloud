from huggingface_hub import HfApi

# --- تنظیمات ---
TOKEN = ""  # توکن خود را اینجا قرار دهید (با hf_ شروع می‌شود)
REPO_ID = "Keyvan1986/Emma-Classification_Model"  # نام دقیق مخزن مدلی که ساختید
# تغییر: اضافه کردن r قبل از گیومه "
LOCAL_FOLDER = r"C:\Users\keyva\MMPL_gpt\Classification Model\final_xlm_r_model_router"


# مثال آدرس ویندوز: "C:\\Users\\Keyvan\\Project\\final_xlm_r_model_router"
# یا اگر پوشه کنار همین فایل است: "./final_xlm_r_model_router"

api = HfApi()

print("🚀 Starting upload to Hugging Face...")

# این دستور تمام محتویات پوشه را به ریشه مخزن آپلود می‌کند
api.upload_folder(
    folder_path=LOCAL_FOLDER,
    repo_id=REPO_ID,
    repo_type="model",
    token=TOKEN
)

print("✅ Upload completed successfully!")
