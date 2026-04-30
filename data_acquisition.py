"""
Data Acquisition Script
Materi: Pertemuan 2 - Data pada Machine Learning
Deskripsi: Skrip ini mendemonstrasikan 4 metode utama dalam akuisisi data (Data Acquisition)
           sebelum masuk ke tahap Data Understanding.
"""

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import mysql.connector

# =====================================================================
# METODE 1: AKUISISI DATA MANUAL (DARI REPOSITORY / LOKAL)
# =====================================================================
def load_manual_data(file_path):
    """
    Memuat data yang sudah diunduh secara manual (misal: via browser dari Kaggle/UCI)
    [span_4](start_span)[span_5](start_span)ke dalam mesin lokal[span_4](end_span)[span_5](end_span).
    """
    print(f"\n[1] Menjalankan Akuisisi Data Manual...")
    try:
        df = pd.read_csv(file_path)
        print(f"Berhasil memuat file: {file_path}")
        return df
    except FileNotFoundError:
        print(f"File {file_path} tidak ditemukan. Pastikan file sudah diunduh secara manual.")
        return None

# =====================================================================
# METODE 2: AKUISISI DATA VIA PUBLIC API (KAGGLE)
# =====================================================================
def download_kaggle_api(dataset_name):
    """
    [span_6](start_span)Mengakses data secara terprogram menggunakan Kaggle API[span_6](end_span).
    Syarat: File kaggle.json harus sudah dikonfigurasi di direktori ~/.kaggle/
    """
    print(f"\n[2] Menjalankan Akuisisi Data via API (Kaggle)...")
    try:
        # [span_7](start_span)Menggunakan command line interface Kaggle untuk download dataset[span_7](end_span)
        os.system(f"kaggle datasets download {dataset_name}")
        
        # [span_8](start_span)Mengekstrak file zip hasil unduhan[span_8](end_span)
        zip_file = f"{dataset_name.split('/')[1]}.zip"
        os.system(f"unzip -o {zip_file}") 
        
        print(f"Dataset {dataset_name} berhasil diunduh dan diekstrak!")
    except Exception as e:
        print(f"Terjadi kesalahan saat mengakses API: {e}")

# =====================================================================
# METODE 3: AKUISISI DATA DENGAN WEB SCRAPING
# =====================================================================
def scrape_web_data(url):
    """
    [span_9](start_span)[span_10](start_span)Mengekstrak data secara langsung dari halaman web (HTML)[span_9](end_span)[span_10](end_span).
    """
    print(f"\n[3] Menjalankan Web Scraping dari {url}...")
    try:
        # [span_11](start_span)Menggunakan requests.get untuk mengakses URL[span_11](end_span)
        response = requests.get(url)
        response.raise_for_status() # Cek apakah request berhasil
        
        # [span_12](start_span)Melakukan parsing HTML menggunakan BeautifulSoup[span_12](end_span)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Ekstraksi tabel data
        table = soup.find('table')
        if table:
            df = pd.read_html(str(table))[0]
            print("Berhasil mengekstrak tabel dari halaman web.")
            return df
        else:
            print("Tidak ditemukan elemen tabel pada halaman tersebut.")
            return None
    except Exception as e:
        print(f"Terjadi kesalahan saat web scraping: {e}")
        return None

# =====================================================================
# METODE 4: AKUISISI DATA DARI RELATIONAL DATABASE (RDB)
# =====================================================================
def fetch_from_rdb(host, user, password, database, query):
    """
    [span_13](start_span)[span_14](start_span)Mengakses data secara terprogram ke basis data relasional (misal: MariaDB/MySQL)[span_13](end_span)[span_14](end_span).
    """
    print(f"\n[4] Menjalankan Akuisisi dari Relational Database ({database})...")
    try:
        # [span_15](start_span)Buka koneksi menggunakan credential RDB[span_15](end_span)
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        
        # [span_16](start_span)Eksekusi SQL query dan muat ke dalam pandas DataFrame[span_16](end_span)
        df = pd.read_sql(query, connection)
        print("Data berhasil diambil dari database.")
        return df
        
    except mysql.connector.Error as err:
        print(f"Error Database: {err}")
        return None
        
    finally:
        # [span_17](start_span)Menutup koneksi database di dalam try-except block[span_17](end_span)
        if 'connection' in locals() and connection.is_connected():
            connection.close()
            print("Koneksi database ditutup.")

# =====================================================================
# BLOK EKSEKUSI UTAMA
# =====================================================================
if __name__ == "__main__":
    print("=== PIPELINE DATA ACQUISITION ===")

    contoh_url = "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)"
    df_scraped = scrape_web_data(contoh_url)

    if df_scraped is not None:
        print(df_scraped.head())
    # [Contoh Eksekusi 1] Manual
    # Pastikan file CSV tersedia di folder kerja.
    # df_manual = load_manual_data("epl-goalScorer(20-21).csv")
    
    # [span_18](start_span)[Contoh Eksekusi 2] API Kaggle (Sesuai contoh PDF)[span_18](end_span)
    # download_kaggle_api("shreyanshkhandelwal/goal-dataset-top-5-european-leagues")
    
    # [Contoh Eksekusi 3] Web Scraping
    # contoh_url = "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)"
    # df_scraped = scrape_web_data(contoh_url)
    
    # [Contoh Eksekusi 4] Relational Database
    # Konfigurasi disesuaikan dengan environment lokal XAMPP (MariaDB)
    # df_db = fetch_from_rdb(
    #     host="localhost",
    #     user="root",
    #     password="",
    #     database="nama_database_kamu",
    #     query="SELECT * FROM nama_tabel LIMIT 10"
    # )