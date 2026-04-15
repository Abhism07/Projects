import os
import bcrypt
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from cryptography.fernet import Fernet
import json
import socket
import http.server
import socketserver
import threading
import shutil
import time
# ✅ FIX 1: Absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VAULT_DIR = os.path.join(BASE_DIR, "vault")
ENC_DIR = os.path.join(VAULT_DIR, "encrypted_files")
KEY_FILE = os.path.join(VAULT_DIR, "key.key")
PASS_FILE = os.path.join(VAULT_DIR, "password.hash")
META_FILE = os.path.join(VAULT_DIR, "metadata.json")


# ---------- METADATA ---------- #

def load_metadata():
    if not os.path.exists(META_FILE):
        return {}
    with open(META_FILE, "r") as f:
        return json.load(f)


def save_metadata(data):
    with open(META_FILE, "w") as f:
        json.dump(data, f)


# ---------- SETUP ---------- #

def setup_vault():
    os.makedirs(ENC_DIR, exist_ok=True)  # ✅ always ensure folder exists
    return not os.path.exists(PASS_FILE)


def save_password(pwd):
    hashed = bcrypt.hashpw(pwd.encode(), bcrypt.gensalt())
    with open(PASS_FILE, "wb") as f:
        f.write(hashed)

    key = Fernet.generate_key()
    with open(KEY_FILE, "wb") as f:
        f.write(key)

    save_metadata({})


# ---------- ENCRYPT / DECRYPT ---------- #

def load_key():
    return open(KEY_FILE, "rb").read()


def encrypt_file(filepath, target_folder):
    key = load_key()
    fernet = Fernet(key)

    with open(filepath, "rb") as f:
        data = f.read()

    encrypted = fernet.encrypt(data)

    filename = os.path.basename(filepath) + ".enc"
    enc_path = os.path.join(target_folder, filename)

    with open(enc_path, "wb") as f:
        f.write(encrypted)

    metadata = load_metadata()
    metadata[enc_path] = filepath
    save_metadata(metadata)

    os.remove(filepath)


def decrypt_to_temp(enc_path):
    key = load_key()
    fernet = Fernet(key)

    with open(enc_path, "rb") as f:
        encrypted = f.read()

    decrypted = fernet.decrypt(encrypted)

    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(decrypted)
    temp.close()

    return temp.name


# ---------- GUI APP ---------- #

class VaultApp:
    def __init__(self, root):
        self.server = None
        self.server_thread = None
        self.root = root
        self.root.title("Secure Vault 🔐")
        self.root.geometry("650x490")

        self.current_folder = ENC_DIR

        if setup_vault():
            self.create_setup_screen()
        else:
            self.create_login_screen()

    def clear_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    # ---------- SETUP ---------- #
    def create_setup_screen(self):
        self.clear_screen()

        tk.Label(self.root, text="Set Vault Password", font=("Arial", 14)).pack(pady=10)

        self.new_pass = tk.Entry(self.root, show="*")
        self.new_pass.pack(pady=5)

        tk.Button(self.root, text="Create Vault", command=self.create_vault).pack(pady=10)

    def create_vault(self):
        pwd = self.new_pass.get()
        if not pwd:
            messagebox.showerror("Error", "Password cannot be empty")
            return

        save_password(pwd)
        messagebox.showinfo("Success", "Vault Created!")
        self.create_login_screen()

    # ---------- LOGIN ---------- #
    def create_login_screen(self):
        self.clear_screen()

        tk.Label(self.root, text="Enter Vault Password", font=("Arial", 14)).pack(pady=10)

        self.password_entry = tk.Entry(self.root, show="*")
        self.password_entry.pack(pady=5)

        tk.Button(self.root, text="Unlock", command=self.check_login).pack(pady=10)

    def check_login(self):
        pwd = self.password_entry.get().encode()

        with open(PASS_FILE, "rb") as f:
            stored = f.read()

        if bcrypt.checkpw(pwd, stored):
            self.create_vault_screen()
        else:
            messagebox.showerror("Error", "Wrong Password")

    # ---------- VAULT ---------- #
    def create_vault_screen(self):
        self.clear_screen()

        self.tree = ttk.Treeview(self.root, show="tree")
        self.tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.tree.bind("<Double-1>", self.open_item)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack()

        tk.Button(btn_frame, text="➕ Add File", command=self.add_file).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="📂 Open File", command=self.open_file).grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="🔁 Restore", command=self.restore_file).grid(row=0, column=2, padx=5)
        tk.Button(btn_frame, text="📁 New Folder", command=self.create_folder).grid(row=0, column=3, padx=5)
        tk.Button(btn_frame, text="⬅ Back", command=self.go_back).grid(row=0, column=4, padx=5)
        tk.Button(btn_frame, text="Start server", command=self.start_server,bg="green",fg="white").grid(row=0, column=5, padx=5)
        tk.Button(btn_frame, text="Stop server", command=self.stop_server,bg="red",fg="white").grid(row=0, column=6, padx=5)

        self.load_files()  # ✅ FIX: THIS WAS MISSING

   
    def start_server(self):
        if self.server:
            messagebox.showinfo("Info", "Server already running!")
            return

        handler = http.server.SimpleHTTPRequestHandler
        self.server = socketserver.TCPServer(("0.0.0.0", 8000), handler)

        def get_local_ip():
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                # doesn't actually connect to internet
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
            finally:
                s.close()
            return ip
        ip = get_local_ip()

        def run_server():
            SERVE_DIR = os.path.join(VAULT_DIR, "temp_serving")

            # clean old temp
            if os.path.exists(SERVE_DIR):
                shutil.rmtree(SERVE_DIR)

            os.makedirs(SERVE_DIR)

            key = load_key()
            fernet = Fernet(key)

            # decrypt all files
            for root, dirs, files in os.walk(self.current_folder):
                for file in files:
                    enc_path = os.path.join(root, file)

                    with open(enc_path, "rb") as f:
                        encrypted = f.read()

                    decrypted = fernet.decrypt(encrypted)

                    # keep folder structure
                    rel_path = os.path.relpath(enc_path, self.current_folder)
                    new_path = os.path.join(SERVE_DIR, rel_path.replace(".enc", ""))

                    os.makedirs(os.path.dirname(new_path), exist_ok=True)

                    with open(new_path, "wb") as f:
                        f.write(decrypted)

            os.chdir(SERVE_DIR)
            self.server.serve_forever()

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

        messagebox.showinfo("Server Started", f"Open on phone:\nhttp://{ip}:8000")
    
    def stop_server(self):
        SERVE_DIR = os.path.join(VAULT_DIR, "temp_serving")
        if self.server:
            for i in range(5):  # retry 5 times
                try:
                    if os.path.exists(SERVE_DIR):
                        shutil.rmtree(SERVE_DIR)
                    return
                except PermissionError:
                    time.sleep(1)  # wait and retry
            self.server.shutdown()
            self.server.server_close()
            self.server = None
            messagebox.showinfo("Stopped", "Server stopped successfully!")
        else:
            messagebox.showinfo("Info", "Server not running")
    # ---------- ICON LOGIC ---------- #
    def get_icon(self, filename, is_folder):
        if is_folder:
            return "📁"

        filename = filename.lower()

        if filename.endswith((".png", ".jpg", ".jpeg", ".gif")):
            return "🖼"
        elif filename.endswith((".mp4", ".mkv", ".avi")):
            return "🎬"
        elif filename.endswith((".pdf", ".docx", ".txt")):
            return "📄"
        else:
            return "📦"

    def load_files(self):
        self.tree.delete(*self.tree.get_children())

        for item in os.listdir(self.current_folder):
            full_path = os.path.join(self.current_folder, item)

            icon = self.get_icon(item, os.path.isdir(full_path))
            self.tree.insert("", "end", text=f"{icon} {item}")

    def open_item(self, event):
        selected = self.tree.focus()
        if not selected:
            return

        name = self.tree.item(selected, "text").split(" ", 1)[1]
        path = os.path.join(self.current_folder, name)

        if os.path.isdir(path):
            self.current_folder = path
            self.load_files()

    def go_back(self):
        if self.current_folder != ENC_DIR:
            self.current_folder = os.path.dirname(self.current_folder)
            self.load_files()

    def create_folder(self):
        name = simpledialog.askstring("Folder", "Enter folder name:")
        if not name:
            return

        path = os.path.join(self.current_folder, name)

        if os.path.exists(path):
            messagebox.showerror("Error", "Folder exists!")
            return

        os.makedirs(path)
        self.load_files()

    def add_file(self):
        path = filedialog.askopenfilename()
        if path:
            encrypt_file(path, self.current_folder)
            self.load_files()

    def open_file(self):
        selected = self.tree.focus()
        if not selected:
            return

        name = self.tree.item(selected, "text").split(" ", 1)[1]
        path = os.path.join(self.current_folder, name)

        if os.path.isdir(path):
            return

        pwd = simpledialog.askstring("Password", "Enter password:", show="*")
        if not pwd:
            return

        with open(PASS_FILE, "rb") as f:
            if not bcrypt.checkpw(pwd.encode(), f.read()):
                messagebox.showerror("Error", "Wrong password!")
                return

        temp = decrypt_to_temp(path)
        os.startfile(temp)

        messagebox.showinfo("Info", "Close file then click OK")

        try:
            os.remove(temp)
        except:
            messagebox.showerror("Error", "Close file first!")

    def restore_file(self):
        selected = self.tree.focus()
        if not selected:
            return

        name = self.tree.item(selected, "text").split(" ", 1)[1]
        enc_path = os.path.join(self.current_folder, name)

        metadata = load_metadata()
        if enc_path not in metadata:
            messagebox.showerror("Error", "Original path not found!")
            return

        original_path = metadata[enc_path]

        key = load_key()
        fernet = Fernet(key)

        with open(enc_path, "rb") as f:
            decrypted = fernet.decrypt(f.read())

        os.makedirs(os.path.dirname(original_path), exist_ok=True)

        with open(original_path, "wb") as f:
            f.write(decrypted)

        os.remove(enc_path)

        del metadata[enc_path]
        save_metadata(metadata)

        self.load_files()


# ---------- MAIN ---------- #

def main():
    root = tk.Tk()
    app = VaultApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()