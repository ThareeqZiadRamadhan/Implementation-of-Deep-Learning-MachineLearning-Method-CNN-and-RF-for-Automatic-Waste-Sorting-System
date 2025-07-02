import os

# Path relatif terhadap Current Working Directory (CWD)
# CWD Anda seharusnya adalah 'C:\Users\harim\Downloads\ani dj\' saat menjalankan skrip
dataset_path_to_test = r'archive'
# Untuk lebih pasti, Anda juga bisa mencoba path absolut (hapus tanda # di baris bawah jika ingin mencoba ini)
# dataset_path_to_test = r'C:\Users\harim\Downloads\ani dj\Garbage classification'

# Dapatkan path absolut untuk memastikan path mana yang sedang diuji
absolute_path = os.path.abspath(os.path.join(os.getcwd(), dataset_path_to_test))
print(f"Current Working Directory (CWD) saat skrip ini dijalankan: {os.getcwd()}")
print(f"Path yang diuji (relatif terhadap CWD jika bukan absolut): '{dataset_path_to_test}'")
print(f"Path absolut yang akan diakses: {absolute_path}\n")

if os.path.exists(absolute_path):
    print(f"STATUS: Direktori '{absolute_path}' DITEMUKAN.")
    if os.path.isdir(absolute_path):
        print(f"STATUS: '{absolute_path}' adalah sebuah direktori.")
        try:
            sub_items = os.listdir(absolute_path)
            print(f"Isi dari '{absolute_path}': {sub_items}")

            if not sub_items:
                print("PERINGATAN: Direktori dataset utama ('{absolute_path}') tampaknya kosong (tidak ada subfolder kategori di dalamnya).")
            else:
                print("\n--- Memeriksa Subfolder Kategori ---")
                all_categories_empty_or_no_images = True
                image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tif', '.tiff'] # Ekstensi gambar umum

                for item in sub_items:
                    item_path = os.path.join(absolute_path, item)
                    if os.path.isdir(item_path):
                        print(f"\n  Subfolder Kategori: '{item}'")
                        category_contents = os.listdir(item_path)
                        if not category_contents:
                            print(f"    PERINGATAN: Subfolder kategori '{item}' KOSONG.")
                        else:
                            image_count = 0
                            non_image_files = []
                            for f_name in category_contents:
                                if os.path.splitext(f_name)[1].lower() in image_extensions:
                                    image_count += 1
                                else:
                                    non_image_files.append(f_name)

                            if image_count > 0:
                                print(f"    INFO: Subfolder kategori '{item}' berisi {image_count} file gambar yang dikenali.")
                                if non_image_files:
                                    print(f"    INFO: Juga terdapat {len(non_image_files)} file/folder lain: {non_image_files[:5]}{'...' if len(non_image_files) > 5 else ''}")
                                all_categories_empty_or_no_images = False
                            else:
                                print(f"    PERINGATAN: Subfolder kategori '{item}' berisi {len(category_contents)} item, TAPI TIDAK ADA FILE GAMBAR yang dikenali (ekstensi: {', '.join(image_extensions)}).")
                                if category_contents:
                                    print(f"    Isi subfolder '{item}': {category_contents[:5]}{'...' if len(category_contents) > 5 else ''}")
                    else:
                        print(f"\n  Item '{item}' di dalam '{absolute_path}' BUKAN direktori (ini tidak diharapkan untuk struktur dataset).")

                if all_categories_empty_or_no_images and sub_items:
                     print("\nKESIMPULAN SEMENTARA: Semua subfolder kategori tampaknya kosong dari file gambar, atau direktori dataset utama tidak memiliki subfolder yang berisi gambar.")

        except PermissionError:
            print(f"ERROR: Tidak ada izin untuk mengakses isi dari '{absolute_path}'. Mohon periksa izin folder.")
        except Exception as e:
            print(f"ERROR saat mencoba membaca isi direktori: {e}")
    else:
        print(f"ERROR: Path '{absolute_path}' ditemukan TETAPI BUKAN sebuah direktori.")
else:
    print(f"ERROR: Direktori '{absolute_path}' TIDAK DITEMUKAN. Pastikan path sudah benar dan folder ada.")

print("\n--- Selesai pengecekan dasar ---")