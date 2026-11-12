# Transcript

Aplikasi Android pribadi untuk transkrip requirement gathering (Indonesia, Inggris, Jawa, sedikit Jepang).

## Uji Gemini Transcribe

1. Simpan API key Gemini sebagai secret `GEMINI_API_KEY` (Settings → Secrets and variables → Actions).
2. Taruh satu rekaman uji (1–3 menit, 2+ pembicara) di folder `samples/`.
3. Opsional: `samples/glossary.txt`, satu istilah per baris.
4. Uji berjalan otomatis setiap ada perubahan di `samples/`, atau jalankan manual dari tab Actions.
5. Hasil tersimpan di `results/<waktu>/report.md`.

Jangan taruh rekaman meeting klien asli di sini.
