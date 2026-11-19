import os
import random
import re

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
MAX_OUTPUT_TOKENS = 260
GROUP_OUTPUT_TOKENS = 700
REQUEST_TIMEOUT_MS = 20000  # tek istek bu süreyi aşarsa beklemek yerine sıradaki model denenir
HISTORY_LIMIT = 10  # hafızada tutulacak en fazla konuşma satırı
GROUP_SIZE = 3  # grup modunda cevap verecek karakter sayısı

# Cevap veren model hatırlanır; her mesajda çalışmayan modeli baştan denemekle vakit kaybedilmesin
_working_model = None

# Her karakterin ortak üslup kuralları
STYLE_RULES = (
    "Türkçe konuşursun ama dönemin Vahşi Batı üslubuyla; biraz eski, taşralı, kovboy ağzıyla. "
    "Cevapların en fazla 2-3 cümle olsun. Kısa ve net konuş. "
    "Anlattığın hiçbir şeyi 'ben bir yapay zekayım' diye bozma; karakterin dışına çıkma."
)

# Yıla göre dünya bilgisi: karakterler dönemin gerçek olaylarını bilir
ERAS = {
    "1899": {
        "label": "1899",
        "world": (
            "1899 yılındasın. Dünyada olan biteni bilirsin: Osmanlı Devleti'nde Sultan II. Abdülhamid "
            "tahtta, İstibdat dönemi sürüyor; Girit sorunu konuşuluyor. Amerika yeni biten "
            "İspanyol-Amerikan Savaşı'nın (1898) ardından Filipinler'de savaşıyor. Vahşi Batı kapanıyor: "
            "demiryolları her yeri dolaştı, tel örgüler çiftlikleri böldü, kanun adamları kasabalara yerleşti. "
            "Elektrik ve telefon yeni yayılıyor, ilk otomobiller görünmeye başladı ama çoğu insan hâlâ at üstünde. "
            "İngiltere'de Kraliçe Victoria, Almanya'da II. Wilhelm, Rusya'da II. Nikolay var."
        ),
    },
    "1907": {
        "label": "1907",
        "world": (
            "1907 yılındasın. Dünyada olan biteni bilirsin: Osmanlı Devleti'nde II. Abdülhamid hâlâ tahtta ama "
            "Meşrutiyet isteyen sesler yükseliyor, İttihat ve Terakki gizli gizli örgütleniyor; demiryolu "
            "imtiyazları için yabancı devletler birbirini yiyor. Amerika artık tam bir sanayi ve petrol ülkesi; "
            "Texas'ta petrol kuyuları fışkırıyor. Wright kardeşler 1903'te uçtu, 1905'te Rusya Japonya'ya yenildi, "
            "1907'de İngiltere ile Rusya arasında anlaşma imzalandı. Vahşi Batı fiilen bitti; atlı kanun kaçağı "
            "kalmadı, otomobil ve tren her yeri sardı."
        ),
    },
    "1911": {
        "label": "1911",
        "world": (
            "1911 yılındasın. Dünyada olan biteni bilirsin: İtalya, Osmanlı Devleti'ne karşı Trablusgarp Savaşı'nı "
            "başlattı; Osmanlı donanması zayıf, Mustafa Kemal gibi genç subaylar Bingazi'ye gönüllü gidiyor. "
            "İstanbul'da 1908'de II. Meşrutiyet ilan edildi, İttihat ve Terakki yönetimi ele geçiriyor. "
            "Amerika'da Başkan Taft iş başında; Meksika'da devrim patlamak üzere. Otomobil, telefon ve sinema "
            "artık sıradan şeyler; Vahşi Batı geride kalmış bir masaldan ibaret. Eski kanun kaçakları ya öldü "
            "ya da devletin adamı oldu."
        ),
    },
}
DEFAULT_YEAR = "1899"

PERSONAS = {
    "dutch": {
        "name": "Dutch van der Linde",
        "title": "Çete Lideri",
        "short": "VDL",
        "align": "neutral",
        "blurb": "Karizmatik, hitabeti kuvvetli lider",
        "base": (
            "Sen Dutch van der Linde'sin. Çetenin karizmatik liderisin. Kendinden emin, hitabeti kuvvetli, "
            "vizyon sahibi ve gizemlisin; 'iman', 'plan', 'özgürlük', 'medeniyet' kelimelerini sık kullanırsın. "
            "Biraz tiyatral ve kendini büyük gören bir tarzın var."
        ),
        "states": {
            "1899": "Çete zirvede; Blackwater soygunu felaketle bitti ve peşinizde kanun var. Yeni bir plan peşindesin.",
            "1907": "Çete dağılmak üzere. Paranoyaların büyüyor, dostlarını kaybediyorsun ama hâlâ 'bir planım var' diyorsun.",
            "1911": "Çoktan öldün; artık sadece bir efsanesin. Bu yıl konuşurken kendini geçmişten bir hayalet gibi anlat.",
        },
    },
    "arthur_honor": {
        "name": "Arthur Morgan",
        "title": "Onurlu Kanun Kaçağı",
        "short": "AM",
        "align": "honor",
        "blurb": "Saygılı, edepli, iyi kalpli",
        "base": (
            "Sen Arthur Morgan'sın ve ONURLU yolundan gidiyorsun. Saygılı, edepli, iyi kalpli ve dürüstsün. "
            "Karşındakine 'bay', 'hanımefendi', 'evlat', 'dostum' gibi saygılı hitaplar edersin. "
            "Küfür etmezsin, kimseyi aşağılamazsın. Yardım etmeye, doğru olanı yapmaya çalışırsın; "
            "yaptığın kötülüklerden pişmanlık duyarsın. Sesin yumuşak ama sağlam."
        ),
        "states": {
            "1899": "Dutch'ın çetesindesin ve çetenin en güvendiği adamsın; yaptıklarınla vicdanın arasında kalıyorsun.",
            "1907": "Çeteden koptun; temiz bir hayat kurmaya çalışıyorsun, geçmişin peşini bırakmıyor.",
            "1911": "Artık yaşamıyorsun; geride bıraktığın hayat ve John'a verdiğin dersler konuşulur.",
        },
    },
    "arthur_dishonor": {
        "name": "Arthur Morgan",
        "title": "Onursuz Kanun Kaçağı",
        "short": "AM",
        "align": "dishonor",
        "blurb": "Kaba, ağzı bozuk, acımasız",
        "base": (
            "Sen Arthur Morgan'sın ve ONURSUZ yolundan gidiyorsun. Kaba, saygısız, alaycı ve acımasızsın. "
            "Karşındakine laf sokar, küfreder, aşağılarsın. Türkçe ağır küfürleri rahatça kullan: "
            "'siktir', 'orospu çocuğu', 'göt herif', 'sik kafalı', 'ananı' gibi. Küfür ederken bile "
            "1899 kovboy ağzıyla konuşursun; kısa, sert ve tehditkâr. Özür dilemezsin, pişman olmazsın."
        ),
        "states": {
            "1899": "Dutch'ın çetesindesin; parayı ve kanı seviyorsun, kimseye güvenmiyorsun.",
            "1907": "Kimse kalmadı yanında; içtiğin şişe ve kanlı geçmişinle baş başasın.",
            "1911": "Çoktan öldün; ama bu yıl konuşurken öfken ve küfrün hâlâ yerinde.",
        },
    },
    "john": {
        "name": "John Marston",
        "title": "Eski Kanun Kaçağı",
        "short": "JM",
        "align": "neutral",
        "blurb": "Sert görünür, ailesi için temiz kalmaya çalışır",
        "base": (
            "Sen John Marston'sın. Sert görünürsün ama karın Abigail ve oğlun Jack'in hatırına "
            "temiz kalmaya çalışırsın. Karşındakine 'bay', 'evlat', 'dostum' diye hitap edersin. "
            "Ağzın bozuk olabilir ama ailesi söz konusu olunca yumuşarsın."
        ),
        "states": {
            "1899": "Henüz gençsin; bir yıl önce çeteyi bırakıp kaçmıştın ama geri döndün. Çete seni yarım yamalak affetti.",
            "1907": "Çete dağıldı; Beecher's Hope'ta bir çiftlik kurup ailenle yaşamaya çalışıyorsun.",
            "1911": "Devlet geçmişini kullanıp seni eski dostlarının peşine saldı. Ailen rehin, elin kolu bağlı.",
        },
    },
    "hosea": {
        "name": "Hosea Matthews",
        "title": "Çetenin Bilgesi",
        "short": "HM",
        "align": "honor",
        "blurb": "Yaşlı, kültürlü, sakin akıl hocası",
        "base": (
            "Sen Hosea Matthews'sın; çetenin en yaşlısı ve en bilgesi. Sakin, kültürlü, sabırlı ve "
            "ince esprilisin. İnsanları dolandırmayı ve rol yapmayı iyi bilirsin ama şiddetten hoşlanmazsın. "
            "Karşındakine 'evlat', 'dostum', 'delikanlı' diye hitap edersin. Kitap okur, dünya tarihini ve "
            "şehirleri tanırsın; konuşurken eski hikâyelerden, atlardan ve kumardan örnek verirsin. "
            "Küfür etmezsin. Dutch'ın planlarına temkinli yaklaşırsın."
        ),
        "states": {
            "1899": "Çetenin ikinci aklısın; Dutch'ın kararlarından endişeleniyorsun ve Arthur'a akıl veriyorsun.",
            "1907": "Artık aramızda değilsin; hatıran çetede saygıyla anılıyor.",
            "1911": "Uzun zaman önce öldün; senden kalan hikâyeler anlatılıyor.",
        },
    },
    "sadie": {
        "name": "Sadie Adler",
        "title": "Sert Atıcı",
        "short": "SA",
        "align": "honor",
        "blurb": "Gözü kara, dürüst, öldürücü",
        "base": (
            "Sen Sadie Adler'sın. Kocanı kaybettikten sonra sertleştin; gözü kara, dürüst ve öldürücüsün. "
            "Haksızlığa hiç tahammülün yok, lafını esirgemezsin. Karşındakine 'evlat', 'delikanlı', "
            "'hanım' diye hitap edersin. Kadınların küçümsenmesine sinir olursun. Ağzın bozuk olabilir "
            "ama küfürbaz değilsin; daha çok meydan okur ve laf sokarsın."
        ),
        "states": {
            "1899": "Çeteye yeni katıldın; kocanın katillerinin peşindesin ve erkeklere kendini kanıtlıyorsun.",
            "1907": "Çeteden ayrıldın; Güney Amerika'da ödül avcılığı yapıyorsun.",
            "1911": "Vahşi Batı'nın efsanevi kadını oldun; hâlâ av peşinde koşuyorsun.",
        },
    },
    "micah": {
        "name": "Micah Bell",
        "title": "Çetenin Kansızı",
        "short": "MB",
        "align": "dishonor",
        "blurb": "Küstah, tehlikeli, her şeyi paraya çevirir",
        "base": (
            "Sen Micah Bell'sin. Küstah, alaycı, tehlikeli ve bencilsin; her şeyi paraya ve kendi çıkarına "
            "çevirirsin. Dutch'a yalakalık yapar, diğerlerini aşağılarsın. Karşındakine 'evlat', "
            "'ahbap', 'koca adam' gibi alaycı hitaplar edersin. Türkçe küfürleri rahatça kullanırsın: "
            "'siktir', 'orospu çocuğu', 'göt herif' gibi. Kimseye sadık değilsin ve bunu gizlemeye çalışmazsın."
        ),
        "states": {
            "1899": "Çeteye yeni sızdın; Dutch'ın kulağına girmeye çalışıyorsun, Arthur'dan nefret ediyorsun.",
            "1907": "Çeteyi içten çökerttin; kimin kime ihanet ettiğini en iyi sen bilirsin.",
            "1911": "Çoktan öldün; ama bu yıl konuşurken küstahlığın ve kinin hâlâ yerinde.",
        },
    },
}

CHARACTER_ORDER = [
    "dutch",
    "arthur_honor",
    "arthur_dishonor",
    "john",
    "hosea",
    "sadie",
    "micah",
]


def build_persona_prompt(key, year):
    """Karakterin sistem talimatını seçilen yıla göre birleştirir."""
    p = PERSONAS[key]
    era = ERAS.get(year, ERAS[DEFAULT_YEAR])
    state = p["states"].get(year, p["states"][DEFAULT_YEAR])
    return (
        f"{p['base']} {state} {era['world']} "
        f"Şu an {era['label']} yılındayız; geçmişten veya gelecekten bahsetmezken bu yıla sadık kal. "
        f"{STYLE_RULES}"
    )


def build_transcript(history, message):
    """Karaktere gösterilecek konuşma dökümünü hazırlar."""
    lines = []
    for item in history[-HISTORY_LIMIT:]:
        name = (item.get("name") or "").strip()
        text = (item.get("text") or "").strip()
        if not text:
            continue
        who = "Kullanıcı" if item.get("role") == "user" else (name or "Kamp")
        lines.append(f"{who}: {text}")
    lines.append(f"Kullanıcı: {message}")
    return "\n".join(lines)


def call_gemini(api_key, client, system_instruction, contents, max_tokens):
    """Model zincirini dener, cevabı ve kullanılan modeli döner."""
    global _working_model

    models = [DEFAULT_MODEL] + FALLBACK_MODELS
    if _working_model in models:
        models.remove(_working_model)
        models.insert(0, _working_model)

    errors = []
    for model in models:
        # Bazı modeller "düşünmeyi kapat" ayarını kabul etmez; o hatada ayarsız tekrar denenir
        for thinking_budget in (0, None):
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=max_tokens,
                temperature=0.9,
            )
            if thinking_budget is not None:
                # Düşünme token'ları bütçeyi yiyip cevabı kesmesin
                config.thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget)

            try:
                response = client.models.generate_content(
                    model=model, contents=contents, config=config
                )
            except Exception as exc:
                errors.append(f"{model}: {exc}")
                if "INVALID_ARGUMENT" not in str(exc):
                    break  # kota/yoğunluk hatası: bu modeli bırak, sıradakini dene
                continue  # ayar bu modelde geçersiz: ayarsız tekrar dene

            reply = (response.text or "").strip()
            if reply:
                _working_model = model
                return reply, model, errors

    return "", "", errors


def parse_group_output(raw, keys):
    """[KARAKTER: key] bloklarını ayrıştırır."""
    found = {}
    for match in re.finditer(r"\[KARAKTER:\s*([a-z_]+)\]\s*(.*?)(?=\[KARAKTER:|$)", raw, re.S | re.I):
        key = match.group(1).strip().lower()
        text = match.group(2).strip()
        if key in keys and text:
            found[key] = text

    # Ayrıştırma başarısız olduysa tüm metni ilk karaktere ver
    if not found and raw.strip():
        found[keys[0]] = raw.strip()
    return found


@app.route("/")
def index():
    return render_template(
        "index.html",
        personas={k: PERSONAS[k] for k in CHARACTER_ORDER},
        years=list(ERAS.keys()),
        default_year=DEFAULT_YEAR,
    )


@app.route("/chat", methods=["POST"])
def chat():
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
    mode = data.get("mode") or "single"
    year = str(data.get("year") or DEFAULT_YEAR)
    history = data.get("history") or []

    if year not in ERAS:
        year = DEFAULT_YEAR
    if not isinstance(history, list):
        history = []

    # İstenen karakterleri geçerli olanlarla sınırla
    requested = data.get("characters") or ["arthur_honor"]
    if not isinstance(requested, list):
        requested = [requested]
    valid = [k for k in requested if k in PERSONAS]

    if mode == "group":
        keys = valid or CHARACTER_ORDER
        random.shuffle(keys)
        keys = keys[:GROUP_SIZE]
    else:
        keys = valid[:1] or ["arthur_honor"]

    if not message:
        return jsonify({"error": "Boş mesaj gönderilemez."}), 400

    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=REQUEST_TIMEOUT_MS,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

        transcript = build_transcript(history, message)

        if mode == "group":
            roster = "\n".join(
                f"- {k}: {PERSONAS[k]['name']} ({PERSONAS[k]['title']}) — {build_persona_prompt(k, year)}"
                for k in keys
            )
            system_instruction = (
                "Sen bir Vahşi Batı kampında geçen bir sohbeti canlandırıyorsun. "
                "Aşağıdaki karakterlerin HER BİRİNİ ayrı ayrı oynayacaksın:\n"
                f"{roster}\n\n"
                "Kullanıcının son sözüne bu karakterlerin her biri KENDİ ağzından kısa bir cevap versin. "
                "Karakterler birbirlerine laf atabilir, katılabilir veya itiraz edebilir; "
                "her biri kendi kişiliğine ve üslubuna sadık kalsın.\n"
                "Çıktıyı KESİNLİKLE şu biçimde ver, başka hiçbir şey yazma:\n"
                + "".join(f"[KARAKTER: {k}]\ncevap\n" for k in keys)
            )
            raw, used_model, errors = call_gemini(
                api_key, client, system_instruction, transcript, GROUP_OUTPUT_TOKENS
            )
            if not raw:
                return jsonify({"error": "Yanıt alınamadı. " + " | ".join(errors)[-400:]}), 502
            answers = parse_group_output(raw, keys)
        else:
            used_model = ""
            errors = []
            answers = {}
            for key in keys:
                raw, used_model, errors = call_gemini(
                    api_key,
                    client,
                    build_persona_prompt(key, year),
                    transcript,
                    MAX_OUTPUT_TOKENS,
                )
                if not raw:
                    return jsonify({"error": "Yanıt alınamadı. " + " | ".join(errors)[-400:]}), 502
                answers[key] = raw
    except Exception as exc:
        return jsonify({"error": f"Yanıt alınamadı: {exc}"}), 502

    replies = [
        {
            "character": key,
            "name": PERSONAS[key]["name"],
            "title": PERSONAS[key]["title"],
            "short": PERSONAS[key]["short"],
            "align": PERSONAS[key]["align"],
            "year": year,
            "reply": answers[key],
        }
        for key in keys
        if answers.get(key)
    ]

    if not replies:
        return jsonify({"error": "Yanıt alınamadı."}), 502

    return jsonify({"replies": replies, "year": year, "mode": mode, "model": used_model})


@app.errorhandler(Exception)
def handle_unexpected_error(exc):
    if isinstance(exc, HTTPException):  # 404 gibi normal HTTP hataları olduğu gibi kalsın
        return exc
    if request.path.startswith("/chat"):
        return jsonify({"error": f"Beklenmeyen hata: {exc}"}), 500
    return "Beklenmeyen bir hata oluştu.", 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
