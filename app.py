import os

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

try:
    from google import genai
    from google.genai import types

    GENAI_AVAILABLE = True
except ImportError:  # paket kurulu değilse uygulama yine de ayağa kalksın
    GENAI_AVAILABLE = False

app = Flask(__name__)

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
# Varsayılan model kota/yoğunluk yüzünden cevap veremezse sırayla denenecek yedekler
FALLBACK_MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.6-flash"]
MAX_OUTPUT_TOKENS = 200
REQUEST_TIMEOUT_MS = 15000  # tek istek bu süreyi aşarsa beklemek yerine sıradaki model denenir

# Cevap veren model hatırlanır; her mesajda çalışmayan modeli baştan denemekle vakit kaybedilmesin
_working_model = None

PERSONAS = {
    "arthur": {
        "name": "Arthur Morgan",
        "title": "Kanun Kaçağı",
        "year": "1899",
        "short": "AM",
        "prompt": (
            "Sen Arthur Morgan'sın. 1899 yılında, Amerikan Vahşi Batısı'nın son günlerinde "
            "yaşayan bir kanun kaçağısın ve Dutch van der Linde çetesinin en güvendiği adamısın. "
            "Sert, sadık ve dürüstsün; lafı uzatmazsın. Karşındakine 'bay', 'evlat', 'dostum' "
            "gibi hitaplar edersin. Türkçe konuşursun ama 1899 Vahşi Batı üslubuyla; biraz eski, "
            "taşralı, kovboy ağzıyla. Cevapların en fazla 2-3 cümle olsun. Kısa ve net."
        ),
    },
    "dutch": {
        "name": "Dutch van der Linde",
        "title": "Çete Lideri",
        "year": "1899",
        "short": "VDL",
        "prompt": (
            "Sen Dutch van der Linde'sin. 1899 yılında Dutch van der Linde çetesinin karizmatik "
            "liderisin. Kendinden emin, hitabeti kuvvetli, vizyon sahibi ve gizemlisin; 'iman', "
            "'plan', 'özgürlük' kelimelerini sık kullanırsın. Türkçe konuşursun ama 1899 Vahşi Batı "
            "üslubuyla, biraz tiyatral ve eski bir dille. Cevapların en fazla 2-3 cümle olsun. "
            "Kısa ve etkileyici."
        ),
    },
    "john": {
        "name": "John Marston",
        "title": "Eski Kanun Kaçağı",
        "year": "1911",
        "short": "JM",
        "prompt": (
            "Sen John Marston'sın. 1911 yılındasın: Dutch van der Linde çetesi çoktan dağıldı, sen "
            "Beecher's Hope'da karın Abigail ve oğlun Jack ile bir çiftlikte yaşıyorsun; ama devlet "
            "geçmişini kullanıp seni eski dostlarının peşine saldı. Sert görünürsün ama karının ve "
            "oğlunun hatırına artık temiz kalmaya çalışıyorsun. Karşındakine 'bay', 'evlat', 'dostum' "
            "diye hitap edersin. Türkçe konuşursun ama 1911 Vahşi Batı üslubuyla; biraz eski, taşralı "
            "ve yorgun bir dille. Cevapların en fazla 2-3 cümle olsun. Kısa ve net."
        ),
    },
}


@app.route("/")
def index():
    return render_template("index.html", personas=PERSONAS)


@app.route("/chat", methods=["POST"])
def chat():
    global _working_model

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return (
            jsonify(
                {
                    "error": "GEMINI_API_KEY ortam değişkeni tanımlı değil. "
                    "Anahtarı ortam değişkeni olarak ayarlayıp uygulamayı yeniden başlatın."
                }
            ),
            500,
        )

    if not GENAI_AVAILABLE:
        return jsonify({"error": "google-genai paketi kurulu değil. 'pip install google-genai' çalıştırın."}), 500

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    character = data.get("character") or "arthur"

    if not message:
        return jsonify({"error": "Boş mesaj gönderilemez."}), 400
    if character not in PERSONAS:
        character = "arthur"

    try:
        # SDK'nın uzun süren otomatik tekrarları kapalı: hata olursa hemen yedek modele geçilir
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT_MS,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

        reply = ""
        used_model = ""
        errors = []

        # Son çalışan model varsa onunla başla, olmazsa varsayılanı dene
        models = [DEFAULT_MODEL] + FALLBACK_MODELS
        if _working_model in models:
            models.remove(_working_model)
            models.insert(0, _working_model)

        for model in models:
            # Bazı modeller "düşünmeyi kapat" ayarını kabul etmez; o hatada ayarsız tekrar denenir
            for thinking_budget in (0, None):
                config = types.GenerateContentConfig(
                    system_instruction=PERSONAS[character]["prompt"],
                    max_output_tokens=MAX_OUTPUT_TOKENS,
                    temperature=0.9,
                )
                if thinking_budget is not None:
                    # Düşünme token'ları 200'lük bütçeyi yiyip cevabı kesmesin
                    config.thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget)

                try:
                    response = client.models.generate_content(
                        model=model, contents=message, config=config
                    )
                except Exception as exc:
                    errors.append(f"{model}: {exc}")
                    if "INVALID_ARGUMENT" not in str(exc):
                        break  # kota/yoğunluk hatası: bu modeli bırak, sıradakini dene
                    continue  # ayar bu modelde geçersiz: ayarsız tekrar dene

                reply = (response.text or "").strip()
                used_model = model
                break

            if reply:
                break

        if not reply:
            return jsonify({"error": "Yanıt alınamadı. " + " | ".join(errors)[-400:]}), 502

        _working_model = used_model
    except Exception as exc:
        return jsonify({"error": f"Yanıt alınamadı: {exc}"}), 502

    return jsonify(
        {
            "reply": reply,
            "character": character,
            "name": PERSONAS[character]["name"],
            "year": PERSONAS[character]["year"],
            "short": PERSONAS[character]["short"],
            "model": used_model,
        }
    )


@app.errorhandler(Exception)
def handle_unexpected_error(exc):
    if isinstance(exc, HTTPException):  # 404 gibi normal HTTP hataları olduğu gibi kalsın
        return exc
    if request.path.startswith("/chat"):
        return jsonify({"error": f"Beklenmeyen hata: {exc}"}), 500
    return "Beklenmeyen bir hata oluştu.", 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
