import os
import sys
import json
import time
import shutil
import gradio as gr
from huggingface_hub import HfApi, snapshot_download

# LlamaIndex Imports
try:
    from llama_index.core import StorageContext, load_index_from_storage
except ImportError:
    pass

# Local Imports setup
bank_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../memory_bank')
sys.path.append(bank_path)

try:
    from build_memory_index import build_memory_index
    from summarize_memory import summarize_memory
except ImportError:
    def build_memory_index(*args, **kwargs): pass
    def summarize_memory(*args, **kwargs): return {}

# ==========================================
# 1. تنظیمات مسیر و هاب (Cloud Config)
# ==========================================

REPO_ID = "Keyvan1986/Emma-memory-storage"
REPO_TYPE = "dataset"
HF_TOKEN = os.environ.get("HF_TOKEN")

# مسیرهای اصلی (Absolute Paths)
BASE_DIR = os.path.abspath(os.getcwd())  # معمولاً /app
MEMORIES_DIR = os.path.join(BASE_DIR, "memories")
MEMORY_INDEX_DIR_NAME = "memory_index"
MEMORY_INDEX_PATH = os.path.join(MEMORIES_DIR, MEMORY_INDEX_DIR_NAME)

MEMORY_FILE_NAME = "update_memory_0512_eng.json"
MEMORY_FILE_PATH = os.path.join(MEMORIES_DIR, MEMORY_FILE_NAME)

# --- مسیر "اشتباه" قدیمی که فایل‌ها آنجا ساخته می‌شوند ---
# لاگ شما نشان داد فایل‌ها اینجا می‌روند: ../memories
LEGACY_OUTSIDE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "memories"))
LEGACY_INDEX_PATH = os.path.join(LEGACY_OUTSIDE_DIR, MEMORY_INDEX_DIR_NAME)

os.makedirs(MEMORIES_DIR, exist_ok=True)
os.makedirs(MEMORY_INDEX_PATH, exist_ok=True)

print(f"📂 Path Configuration:")
print(f"   - App Dir (Target): {MEMORY_INDEX_PATH}")
print(f"   - Builder Dir (Source): {LEGACY_INDEX_PATH}")

# ==========================================
# 2. تابع انتقال فایل (حیاتی برای حل مشکل شما)
# ==========================================

def sync_legacy_files_to_app():
    """
    این تابع بررسی می‌کند که آیا ایندکس‌ها در مسیر ../memories ساخته شده‌اند یا نه.
    اگر آنجا باشند، آن‌ها را به داخل /app/memories کپی می‌کند تا آپلود شوند.
    """
    if os.path.exists(LEGACY_INDEX_PATH):
        print(f"🕵️ Detected files in outside path: {LEGACY_INDEX_PATH}")
        try:
            # کپی کردن تمام محتویات از بیرون به داخل پوشه قابل آپلود
            shutil.copytree(LEGACY_INDEX_PATH, MEMORY_INDEX_PATH, dirs_exist_ok=True)
            print(f"✅ MOVED files from '{LEGACY_INDEX_PATH}' to '{MEMORY_INDEX_PATH}' for upload.")
        except Exception as e:
            print(f"⚠️ Error moving legacy files: {e}")
    else:
        # شاید فایل‌ها همین الان در جای درست باشند
        pass

# ==========================================
# 3. توابع همگام‌سازی ابری
# ==========================================

def push_to_hub():
    if not HF_TOKEN:
        return

    try:
        # قبل از آپلود، مطمئن می‌شویم فایل‌ها در جای درست هستند
        sync_legacy_files_to_app()

        api = HfApi(token=HF_TOKEN)
        
        # 1. آپلود JSON
        if os.path.exists(MEMORY_FILE_PATH):
            api.upload_file(
                path_or_fileobj=MEMORY_FILE_PATH,
                path_in_repo=MEMORY_FILE_NAME,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                commit_message=f"Auto-save JSON: {time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        
        # 2. آپلود پوشه ایندکس
        if os.path.exists(MEMORY_INDEX_PATH):
            # بررسی تعداد فایل‌های واقعی
            files_count = sum([len(files) for r, d, files in os.walk(MEMORY_INDEX_PATH)])
            print(f"📂 Preparing upload. Found {files_count} files in {MEMORY_INDEX_PATH}")

            # ترفند Force Update
            with open(os.path.join(MEMORY_INDEX_PATH, "last_sync_log.txt"), "w") as f:
                f.write(f"Sync triggered at: {time.time()}")

            print("☁️ Uploading Index folder...")
            api.upload_folder(
                folder_path=MEMORY_INDEX_PATH,
                path_in_repo=MEMORY_INDEX_DIR_NAME, 
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                commit_message=f"Auto-save Indices: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                ignore_patterns=[".DS_Store", "*.git*"]
            )
            print(f"✅ Sync Complete.")

    except Exception as e:
        print(f"❌ Cloud Sync Error: {e}")


def pull_from_hub():
    if not HF_TOKEN: return

    try:
        print(f"📥 Pulling data from Hub...")
        # دانلود JSON
        try:
            snapshot_download(
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                local_dir=MEMORIES_DIR,
                allow_patterns=[MEMORY_FILE_NAME],
                force_download=True
            )
        except: pass

        # دانلود پوشه ایندکس
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            local_dir=MEMORIES_DIR,
            allow_patterns=f"{MEMORY_INDEX_DIR_NAME}/**",
            force_download=True
        )
        print("✅ Data Downloaded.")
        
        # برعکس: اگر برنامه در ../memories دنبال فایل می‌گردد، باید فایل‌های دانلود شده را آنجا هم کپی کنیم
        # تا LlamaIndex بتواند آن‌ها را بخواند (اگر مسیرش نسبی است)
        if os.path.exists(MEMORY_INDEX_PATH):
            try:
                os.makedirs(LEGACY_INDEX_PATH, exist_ok=True)
                shutil.copytree(MEMORY_INDEX_PATH, LEGACY_INDEX_PATH, dirs_exist_ok=True)
                print(f"✅ Mirroring downloaded files to legacy path: {LEGACY_INDEX_PATH}")
            except Exception as e:
                print(f"⚠️ Mirroring error: {e}")

    except Exception as e:
        print(f"⚠️ Cloud Download Error: {e}")

# ==========================================
# 4. توابع اصلی I/O
# ==========================================

def save_memory(data):
    try:
        os.makedirs(MEMORIES_DIR, exist_ok=True)
        temp_file = MEMORY_FILE_PATH + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        os.replace(temp_file, MEMORY_FILE_PATH)
        push_to_hub()
    except Exception as e:
        print(f"❌ Save Error: {e}")

def load_memory():
    pull_from_hub()
    if not os.path.exists(MEMORY_FILE_PATH):
        return {}
    try:
        with open(MEMORY_FILE_PATH, 'r', encoding='utf-8') as f:
            return json.loads(f.read())
    except:
        return {}

# ==========================================
# 5. منطق حافظه (Memory Logic)
# ==========================================

def enter_name_llamaindex(name, memory_data, data_args, update_memory_index=True):
    is_new_user = False
    if name not in memory_data:
        memory_data[name] = {"name": name, "sessions": [], "episodic_memory": [], "semantic_memory": {}, "profile": {}}
        is_new_user = True
        save_memory(memory_data)
    
    user_memory = memory_data[name]

    # سعی می‌کنیم فایل‌ها را در هر دو مسیر (جدید و قدیم) چک کنیم
    # مسیر صحیح (که دانلود شده)
    correct_base_dir = os.path.join(MEMORY_INDEX_PATH, "llamaindex", name)
    # مسیر قدیمی (که ممکن است بیلدر آنجا بسازد)
    legacy_base_dir = os.path.join(LEGACY_INDEX_PATH, "llamaindex", name)

    # بررسی وجود فایل در مسیر صحیح
    indices_exist = (
        os.path.exists(os.path.join(correct_base_dir, "sessions_index.json")) and 
        os.path.exists(os.path.join(correct_base_dir, "episodic_memory_index.json"))
    )

    if update_memory_index or not indices_exist:
        try:
            print(f"⚙️ Building indices for {name}...")
            # این تابع احتمالا در ../memories می‌سازد
            build_memory_index(memory_data, data_args, name=name)
            
            # بلافاصله فایل‌ها را به مسیر درست منتقل می‌کنیم و آپلود می‌کنیم
            sync_legacy_files_to_app()
            push_to_hub()
            
        except Exception as e:
            print(f"Warning building index: {e}")

    # بارگذاری (لودینگ)
    # اولویت با مسیری است که فایل دارد. اگر دانلود شده باشد، در correct_base_dir است.
    # اگر همین الان ساخته شده باشد و کپی شده باشد، باز هم در correct_base_dir است.
    load_dir = correct_base_dir if os.path.exists(correct_base_dir) else legacy_base_dir

    sessions_memory = None
    episodic_memory = None
    semantic_memory = None

    def load_idx(path):
        try:
            return load_index_from_storage(StorageContext.from_defaults(persist_dir=path))
        except: return None

    if os.path.exists(load_dir):
        sessions_memory = load_idx(os.path.join(load_dir, "sessions_index")) # ممکن است نام پوشه فرق کند، کد اصلی چک شود
        # معمولا json ذخیره نمیشود، بلکه پوشه است. اگر فایل json است:
        if not sessions_memory:
             # LlamaIndex معمولا پوشه میسازد. اگر فایل json است لاجیک فرق میکند
             # اما طبق لاگ شما: Saved ... to .../episodic_memory (بدون پسوند، یعنی پوشه)
             pass

    # تلاش استاندارد برای لود:
    # توجه: در لاگ شما مسیرها .../episodic_memory بود (پوشه).
    s_path = os.path.join(load_dir, "sessions_index")
    e_path = os.path.join(load_dir, "episodic_memory") # طبق لاگ شما
    m_path = os.path.join(load_dir, "semantic_memory") # طبق لاگ شما

    # چک کردن فایل json اگر پوشه نبود (fallback)
    if not os.path.exists(e_path): e_path += "_index.json"
    if not os.path.exists(m_path): m_path += "_index.json"

    if os.path.exists(s_path): sessions_memory = load_idx(s_path)
    if os.path.exists(e_path): episodic_memory = load_idx(e_path)
    if os.path.exists(m_path): semantic_memory = load_idx(m_path)

    return f"Welcome {name}!", user_memory, sessions_memory, episodic_memory, semantic_memory


def summarize_memory_event_personality(data_args, memory_data, user_name):
    save_memory(memory_data)
    try:
        updated_memory = summarize_memory(MEMORY_FILE_PATH, user_name, language='en')
        # بعد از خلاصه سازی هم ممکن است ایندکس تغییر کند
        sync_legacy_files_to_app()
        push_to_hub()
        return updated_memory.get(user_name, {})
    except Exception as e:
        print(f"Error summarize: {e}")
        return memory_data.get(user_name, {})
