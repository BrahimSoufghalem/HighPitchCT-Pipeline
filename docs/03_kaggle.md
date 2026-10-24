# دليل التشغيل الذكي على Kaggle (`2× T4` + `30 GB RAM` + سقف `18 GiB`)

يوثّق هذا الملف كيفية عمل المشروع بذكاء داخل بيئة **Kaggle Notebooks** للتجريب والتحقق السريع على عينة مصغّرة، وكيف ينتقل الكود نفسه دون تغيير سطر برمجي واحد إلى خادم متطور لتدريب كامل البيانات.

---

## 1. تحليل كفاية موارد Kaggle الفعلية لمشروعنا

| المورد في Kaggle (`GPU T4 x2`) | القيمة الفعلية | كيف يتعامل معها مشروعنا؟ |
| :--- | :--- | :--- |
| **معالجات الرسوميات (VRAM)** | بطاقتان منفصلتان `Tesla T4` ($2 \times 16\text{ GB VRAM}$) | كافية جدًا: شبكتا `Residual U-Net` وطبقة `Differentiable FBP` ثنائية الأبعاد (`2D`) مع `AMP FP16`، ومحاكاة `LEAP` لحجم مريض واحد تستهلك $< 2\text{ GB VRAM}$ |
| **الذاكرة العشوائية (System RAM)** | $\sim 29\text{–}30\text{ GB}$ | كافية جدًا بشرط المعالجة المجزأة (`Chunked / Streaming` + `memmap`)؛ سلسلة صدر خام كاملة تزن $\sim 2.5\text{ GB}$ في `float32`، فلن نحتفظ بالخام والمسطح والسينوغرام معًا في الذاكرة |
| **قرص الإخراج الدائم (`/kaggle/working`)** | حدّ أقصى صارم $\sim 19.5\text{–}20\text{ GB}$ | **القيد الأهم:** يفعّل الكود تلقائيًا سقف `18.0 GiB` صارمًا ويترك هامش أمان $2\text{ GB}$ للـ checkpoints والتقارير |
| **القرص المؤقت (`/tmp`)** | منفصل عن حصة `/kaggle/working` | ينزّل الكود ملفات `.zip` و`.zip.part` داخل `/tmp/highpitchct-incoming` ثم يحذفها بعد فك الضغط والتحقق من `SHA-256`، فلا تضيع مساحة `/kaggle/working` في تكرار الأرشيف! |
| **قرص الإدخال (`/kaggle/input`)** | للقراءة فقط (يسمح بـ $> 100\text{ GB}$) | يرفض الكود الكتابة فيه صراحةً، لكنه يدعم فهرسته وقراءته مباشرةً عبر `--scan-dir /kaggle/input/...` |
| **مهلة الجلسة (Session Timeout)** | حتى $12$ ساعة متواصلة | جميع المراحل تحفظ سجلات `JSON` وملفات استئناف، فلا يضيع أي تقدم عند انتهاء الجلسة |

---

## 2. كم مريضًا يتسع داخل سقف الـ `18.0 GiB` على Kaggle؟

من القياس الفعلي للمريض **`C012`**:
- ملف الإسقاط الواحد يزن **`96.8 KB`**.
- سلسلة `projection_full` لفحص صدر كامل ($\sim 13{,}000$ ملف) تزن **$\approx 1.25\text{–}1.35\text{ GB}$**.
- سلسلة `reconstruction_published` (`351` شريحة) تزن **$\approx 0.18\text{ GB}$**.
- باستبعاد سلسلتي الجرعة المنخفضة (`projection_low` و`reconstruction_low`)، يزن المريض الكامل **$\approx 1.45\text{ GB}$ فقط**!
- بالتالي يتسع سقف **`18.0 GiB`** لنحو **$8$ إلى $10$ مرضى كاملين** (مثلًا: $6$ تدريب + $1$ تحقق + $2$ اختبار)، وهو حجم مثالي لإثبات عمل خط المعالجة والتدريب الثنائي المجال كاملًا على Kaggle قبل نقله إلى بيئة أكبر!

---

## 3. المقارنة التلقائية بين وضعي `kaggle` و`full`

يكتشف `resolve_environment_profile()` البيئة تلقائيًا دون أي تدخل منك:

| الإعداد | عند اكتشاف Kaggle (`profile = kaggle`) | على الحاسوب أو الخادم المتطور (`profile = full`) |
| :--- | :--- | :--- |
| `data_root` الافتراضي | `/kaggle/working/highpitchct-data` | `./data` |
| `incoming_root` لملفات `.zip` | `/tmp/highpitchct-incoming` | `<data_root>/incoming` |
| `max_disk_gb` | **`18.0 GiB`** (يتوقف بأمان عند بلوغه) | **`None`** (ينزّل كامل قائمة المرضى بلا سقف) |
| حذف `.zip` بعد الاستخراج | **`True`** (بعد توثيق `SHA-256`) | **`False`** (يحتفظ بنسخة الأرشيف الخام) |

ويمكنك في أي وقت فرض أي وضع يدويًا لأغراض الاختبار عبر `--profile kaggle` أو `--profile full` أو `--max-disk-gb 10`.

---

## 4. خطوات التشغيل العملية على Kaggle (خطوة بخطوة)

### الخطوة 1: إعدادات دفتر Kaggle قبل التشغيل
1. في الشريط الجانبي الأيمن، فعّل **Internet = On**.
2. لأن المستودع `HighPitchCT-Pipeline` خاص (*Private*)، افتح قائمة **Add-ons $\rightarrow$ Secrets** في أعلى محرر Kaggle، وأضف سرًا جديدًا باسم:
   - **Label:** `GITHUB_TOKEN`
   - **Value:** توكن GitHub الشخصي الخاص بك (`ghp_...`)
   ثم فعّل مربع الاختيار بجانبه لهذا الدفتر. *(بهذه الطريقة يبقى التوكن مشفرًا وآمنًا ولا يظهر أبدًا داخل كود الدفتر).*
3. في **Stage 0** (التنزيل والفهرسة فقط)، يمكنك إبقاء **Accelerator = None** كي لا تستهلك حصة الـ GPU الأسبوعية ($30$ ساعة)، وعند اختبار المراحل التي تحتاج GPU اختر **GPU T4 x2**.

### الخطوة 2: استنساخ المستودع وفحص البيئة (الخلية الأولى)
```python
import os
from pathlib import Path
from kaggle_secrets import UserSecretsClient

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
repo_dir = Path("/kaggle/working/HighPitchCT-Pipeline")
if not repo_dir.exists():
    rc = os.system(
        f"git clone https://{token}@github.com/BrahimSoufghalem/HighPitchCT-Pipeline.git {repo_dir}"
    )
    assert rc == 0, "فشل استنساخ المستودع؛ تأكد من تفعيل Internet وصلاحية GITHUB_TOKEN"

%cd /kaggle/working/HighPitchCT-Pipeline
!python -m pip install -q -e .
!python -m unittest discover -s tests -v
!hpct env
```
تأكد أن مخرجات `!hpct env` تعرض:
- `"kaggle": true`
- `"profile": "kaggle"`
- `"max_disk_gb": 18.0`
- `"incoming_root": "/tmp/highpitchct-incoming"`

### الخطوة 3: تجربة المريض `C012` الجاهز مباشرةً (الخلية الثانية)
```bash
!hpct validate-manifest --series-file configs/pilot_c012.csv
!hpct download --series-file configs/pilot_c012.csv --workers 2 --retries 5
!hpct index
```

### الخطوة 4: أو اكتشاف وتنزيل عدة حالات صدر حتى سقف `18 GiB` تلقائيًا
```bash
# يكتشف مثلاً 20 حالة صدر من TCIA، وعند التنزيل على Kaggle سيتوقف ذكيًا عند ~18 GiB دون امتلاء القرص!
!hpct discover-tcia \
  --collection LDCT-and-Projection-data \
  --body-part CHEST \
  --roles projection_full,reconstruction_published \
  --max-cases 20 \
  --output /kaggle/working/chest_20_cases.csv

!hpct download --series-file /kaggle/working/chest_20_cases.csv --workers 2 --retries 5
!hpct index
```

### الخطوة 5: استعراض جدول الفيزياء والتدقيق في Kaggle
```python
import pandas as pd

inv = pd.read_csv("/kaggle/working/highpitchct-data/manifests/series_inventory.csv")
display(inv[[
    "series_description",
    "dicom_kind",
    "file_count",
    "total_mb",
    "ffs_mode",
    "axial_span_mm",
    "estimated_table_feed_mm_per_rot",
    "estimated_normalized_pitch",
    "pitch_fit_rmse_mm",
    "validation_status",
]])
```
