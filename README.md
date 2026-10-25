# HighPitchCT-Pipeline

إطار بحثي وهندسي محكم وقابل للتدقيق لإعادة بناء وتقييم نموذج **ثنائي المجال (Dual-Domain Deep Learning)** لإعادة بناء التصوير المقطعي الحلزوني مرتفع الـ Pitch (**High-Pitch Helical CT**) انطلاقًا من بيانات الإسقاط الخام في مجموعة **TCIA (`LDCT-and-Projection-data`)**، ومُصمَّم خصيصًا لمذكرة ماستر في **الفيزياء الطبية** وفق نظام بوابات المراحل الصارم (**Stage-Gate**).

> **الإصدار الحالي:** `v0.3.1` — **المرحلة 0 (Stage 0) مكتملة ومُسرَّعة**: اكتشاف السلاسل من واجهة TCIA، **تنزيل متوازٍ فائق السرعة عبر جلسات `requests.Session` المجمّعة (`Keep-Alive` + كتل تدفق `4 MiB`) مع شريط تقدم حيّ `tqdm` وتقرير سرعة `download_report.csv`**، التنزيل الذكي الواعي ببيئة التشغيل (**سقف صارم $\le 18\text{ GiB}$ وتخزين مؤقت في `/tmp` تلقائيًا على Kaggle** مقابل تنزيل كامل غير مقيد على الحواسيب والخوادم)، حساب `SHA-256` لحظيًا أثناء التدفق واستخراج ذري بتمريرة واحدة مع فحص `CRC-32`، وفهرسة متوازية لرؤوس `DICOM-CT-PD v4` (`--workers 8`).

---

## 1. جدول القرارات العلمية والهندسية المعتمدة (المرجع الدستوري للمشروع)

| المحور | القرار المعتمد | المبرر الفيزيائي والهندسي |
| :--- | :--- | :--- |
| **هوية البحث** | إعادة تنفيذ علمية مستقلة وموثّقة بالكامل مستلهمة من ملصق AAPM 2026 دون الاعتماد على مراسلة المؤلفين | ضمان الاستقلالية العلمية وتوثيق كل فرضية ومعادلة بدقة تصلح للمناقشة الأكاديمية |
| **منهجية العمل** | نظام **Stage-Gate** صارم: كود + توثيق تفصيلي مملّ + 4 مستويات اختبار (Unit / Synthetic / Integration / Real-data) لكل مرحلة | منع تراكم الأخطاء الهندسية وعدم الانتقال لأي مرحلة تالية قبل اجتياز بوابة القبول |
| **إدارة الـ Backends** | تحديد صريح للـ Backend في الإعدادات **ومنْع أي انتقال صامت (`No silent fallback`)** | ضمان قابلية تكرار التجارب وعدم خلط خوارزميات إعادة بناء مختلفة بين المرضى |
| **الوعي الذكي بالبيئة** | اكتشاف تلقائي للبيئة (`kaggle` مقابل `full`): على Kaggle يلتزم بسقف **$\le 18.0\text{ GiB}$** في `/kaggle/working` ويضع الـ `.zip` المؤقت في `/tmp` ويحذفه بعد الاستخراج؛ وخارج Kaggle ينزّل كامل القائمة | تمكين التجريب الآمن والسريع لـ $10\text{–}12$ مريضًا على Kaggle ($2\times\text{T4}$)، ثم تشغيل الكود نفسه دون تعديل على خادم أكبر لكامل البيانات |
| **قراءة DICOM-CT-PD** | فكّ ثنائي صريح للوسوم الخاصة (`VR=UN`) وفق دليل `DICOM-CT-PD v4`، ترتيب وفحص استمرارية الزاوية و$z_0$، وتحويل المصفوفة بـ `pixel_array.T` (`order='F'`) إلى `(64, 736)` دون تكرار اللوغاريتم (`LogFlag=YES`) | ملفات TCIA مكتوبة بـ `Implicit VR Little Endian` وبترتيب أعمدة MATLAB (`Rows=736, Columns=64`)؛ أي قراءة بـ `order='C'` تخلط الصفوف بالأعمدة |
| **تقدير الـ Pitch والهندسة** | حساب ديناميكي لكل سلسلة عبر انحدار خطي لـ $z_0(\phi_{\text{unwrap}})$ لاستخراج `table_feed_mm_per_rot` و`normalized_pitch` واتجاه الطاولة | عدم افتراض ثبات الـ pitch عند `0.9` أو `0.6` أو ربطه بنوع العضو التشريحي |
| **البؤرة المتحركة (`FFS`)** | قراءة وتطبيق المركبات الثلاث $(\rho_0+\Delta\rho,\; \phi_0+\Delta\phi,\; z_0+\Delta z)$ لكل view | في جهاز `SOMATOM Definition AS+` حتى عند `FFSZ` تنزاح البؤرة شعاعيًا بـ $\Delta\rho = +5.45\text{ mm}$ بسبب ميل المصعد ($\approx 6.9^\circ$) |
| **الربط بين الصور والإسقاطات** | استخدام تسلسل `ReferenceRawDataSequence (0008,9121)` $\rightarrow$ `(0008,1115)` $\rightarrow$ `(0020,000E)` | في بيانات TCIA يختلف كلٌّ من `StudyInstanceUID` و`FrameOfReferenceUID` بين الصور المنشورة والإسقاطات الخام |
| **Curved-to-Flat & SSRB** | تحويل متجهي سريع مع `measured_mask` صريح وإلغاء `mode='nearest'` خارج الكاشف، واختيار القياس الأقرب لمركز الكاشف ($\arg\min_k \|v_k^\star\|$)، وفصل `Extrapolation` كمرحلة مستقلة | تمثيل نقص البيانات الفعلي الناتج عن الـ high-pitch بأمانة بدل إخفائه بتكرار أطراف الكاشف |
| **المرجع (`Reference Reconstruction`)** | إعادة بناء $x_{\text{reference}}$ من الإسقاطات الخام كاملة الجرعة عند الـ pitch الأصلي لكل فحص عبر خط `SSRB + Short-scan + Parker + FBP` الخاص بنا | منع الشبكة من تعلم تقليد مرشح إعادة البناء التجاري (`B50f` مثلًا) وإزاحات إحداثياته، وحصر التعلم في إزالة تشوهات الـ high-pitch |
| **محاكاة ITF وRDC والتعلم** | تحويل $x_{\text{reference}}$ إلى $\mu(x,y,z)$ وتوليد إسقاطات أمامية حلزونية (`LEAP`) لحالات ITF/RDC، وتدريب شبكتي Residual U-Net مرتبطتين عبر `Differentiable FBP` بخسارة $\mathcal{L}_1 + \mathcal{L}_\nabla$ | مطابقة الفيزياء الفعلية لمسار الطاولة وتطبيق البنية الرياضية الموصوفة في ملصق AAPM 2026 |

---

## 2. تفاصيل المرحلة الحالية: Stage 0 (`v0.3.1`) — البنية والتسريع الفائق

تتكون المرحلة 0 من تسع وحدات برمجية متكاملة داخل `src/highpitchct/` دون تحميل أي مصفوفات بكسل في الذاكرة العشوائية:

1. **`runtime.py` — الكشف الذكي عن البيئة وحارس ميزانية القرص (`DiskBudgetGuard`):**
   - يكتشف تلقائيًا ما إذا كان الكود يعمل داخل **Kaggle** (عبر `KAGGLE_KERNEL_RUN_TYPE` أو وجود `/kaggle/working` و`/kaggle/input`) أم على جهاز/خادم عادي (`full`).
   - في وضع **`kaggle`**:
     - مسار البيانات الدائم: `/kaggle/working/highpitchct-data`.
     - مسار التنزيل المؤقت لملفات `.zip` و`.zip.part`: `/tmp/highpitchct-incoming` (خارج حصة الـ $20\text{ GB}$ الخاصة بـ `/kaggle/working` حتى لا يتضاعف استهلاك القرص أثناء فك الضغط).
     - سقف الميزانية الصارم: **`18.0 GiB`** (`19,327,352,832` بايت). يفحص `DiskBudgetGuard` المساحة المستخدمة بسرعة فائقة عبر `os.scandir` مع دعم الحجز الذري المتزامن (`try_update_reservation`) عبر جميع خيوط التنزيل؛ وإذا كان المريض التالي سيتجاوز سقف $18.0\text{ GiB}$، يتوقف التنزيل بأمان تام ويسجل `"status": "skipped_quota_reached"`.
   - في وضع **`full`** (حاسوب محلي أو خادم متطور):
     - يُلغى سقف الـ $18\text{ GiB}$ (`max_disk_bytes = None`)، فينزّل الكود جميع السلاسل المفعّلة في ملف الـ CSV بالكامل وبنفس الأمر تمامًا.

2. **`tcia_client.py` و`safe_zip.py` و`progress.py` — التنزيل المتوازي فائق السرعة وشريط التقدم الحيّ:**
   - **جلسات `requests.Session` محلية لكل خيط (`Thread-Local Connection Pooling`):** إعادة استخدام اتصالات `HTTPS Keep-Alive` مع `HTTPAdapter` وسياسة إعادة محاولة ذكية (`Retry` على رموز `[429, 500, 502, 503, 504]`) وكتل تدفق كبيرة بحجم **`4 MiB`** (`DEFAULT_CHUNK_BYTES = 4,194,304`).
   - **مجمّع خيوط موحّد عابر للمرضى (`Unified Multi-Patient Worker Pool`):** استعلام متوازٍ مسبق عن أحجام السلاسل (`/getSeriesMetaData`) في أجزاء من الثانية، ثم حجز حتمي لحصة المرضى المقبولين ضمن سقف الـ $18\text{ GiB}$، يلي ذلك تنزيل جميع السلاسل المقبولة بالتوازي الكامل (`--workers 4` افتراضيًا، وقابل للرفع إلى `6` أو `8`) بدل الانتظار التسلسلي لمريض تلو الآخر.
   - **إلغاء الاختناقات الثلاثية بعد التنزيل (توفير $\sim 45\text{–}70$ ثانية لكل مريض):**
     1. **بصمة `SHA-256` أثناء التدفق (`Streaming Hash`):** تُحسب لحظيًا على كتل الشبكة أثناء كتابتها في `.part` (مع دمج الجزء السابق عند الاستئناف `HTTP 206`) دون إعادة قراءة ملف بحجم `1.3 GB` من القرص.
     2. **استخراج ذري بتمريرة واحدة مع فحص `CRC-32` لحظي (`safe_extract_zip_with_stats`):** بدل فك ضغط `13,446` ملفًا مرتين (`testzip()` ثم `extract`) واستدعاء `40,000+` نداء نظام (`resolve` / `mkdir` / `chmod`)، يتحقق الكود من خلو جميع المسارات من `Zip Slip` ومن سعة القرص في أقل من $5\text{ ms}$، ثم يستخرج الملفات دفعةً واحدة داخل مجلد شقيق مؤقت `.extracting_*` بينما تتحقق مكتبة `zipfile` تلقائيًا من `CRC-32` لكل ملف أثناء القراءة، ثم يبدّل المجلد ذريًا (`os.replace`) عند النجاح التام أو يمسحه بالكامل عند أي تلف.
   - **شريط تقدم حيّ (`tqdm`) وتقرير سرعة جدولي (`download_report.csv`):** يعرض نسبة التقدم المئوية `%`، الحجم المنزّل بالميغابايت/الغيغابايت، السرعة اللحظية `MiB/s`، والوقت المتبقي `ETA`، مع رسائل حالة لحظية (`⬇` بدء، `📦` فك ضغط، `✅` اكتمال مع السرعة، `⏭` تخطي)، ويولّد ملف `manifests/download_report.csv` تلقائيًا.

3. **`ctpd_tags.py` و`index_dicom.py` — الفهرسة المتوازية لرؤوس `DICOM-CT-PD v4` وتدقيق السلاسل:**
   - قراءة متوازية لرؤوس الملفات فقط (`stop_before_pixels=True` عبر `ThreadPoolExecutor` بـ `--workers 8` افتراضيًا) مع شريط تقدم حيّ `tqdm` يعرض سرعة الفهرسة (`file/s`) ومخرجات حتمية مرتبة 100%.
   - فكّ ثنائي دقيق للوسوم الخاصة ذات `VR=UN` وفق الجداول الرسمية لدليل `DICOM-CT-PD Version 4`.
   - حساب ديناميكي لكل سلسلة إسقاط خام: فكّ التفاف الزوايا $\phi_{\text{unwrap}}$، ومواءمة خطية $z_0 = a\,\phi_{\text{unwrap}} + b$ لحساب إزاحة الطاولة $F = 2\pi|a|$، التغطية $W_{\text{iso}}$، والـ Pitch الفعلي $p = F / W_{\text{iso}}$ مع $\text{RMSE}$.
   - ربط الصور المنشورة بسلاسل الإسقاط الخام حتميًا عبر `(0008,9121)` وكشف أي بتر محوري (`WARNING_TRUNCATED_PROJECTION_SERIES`).

---

## 3. بنية المستودع والمخرجات

```text
HighPitchCT-Pipeline/
├── configs/
│   ├── pilot_c012.csv                # قائمة موثّقة للمريض C012 (13,446 إسقاطًا كامل الجرعة + 351 صورة منشورة)
│   └── series.example.csv            # قالب توضيحي لملف السلاسل
├── docs/
│   ├── 00_project_overview.md        # النظرة العامة وخارطة المراحل التسع
│   ├── 01_data_download.md           # توثيق تفصيلي للتنزيل فائق السرعة وحارس الـ 18 GiB وشريط التقدم
│   ├── 02_data_organization.md       # توثيق تفصيلي لفك وسوم DICOM-CT-PD v4 وحساب الـ Pitch والتدقيق
│   ├── 03_kaggle.md                  # دليل تشغيل Kaggle (T4 x2 + 30GB RAM + سقف 18 GiB + Kaggle Secrets)
│   ├── validation_protocol.md        # بروتوكول التحقق وبوابات القبول الصارمة
│   └── decisions/                    # سجلات القرارات المعمارية (ADR-0001 إلى ADR-0004)
├── notebooks/
│   └── 00_kaggle_stage0.ipynb        # دفتر جاهز للتشغيل المباشر على Kaggle
├── src/highpitchct/
│   ├── cli.py                        # واجهة الأوامر الموحدة hpct
│   ├── ctpd_tags.py                  # مفكك الوسوم الثنائية الخاصة وفق DICOM-CT-PD v4 ورابط (0008,9121)
│   ├── index_dicom.py                # الفهرسة المتوازية وتدقيق اكتمال السلاسل وتقدير الـ Pitch
│   ├── layout.py                     # مسارات التخزين الحتمية وفصل incoming المؤقت عن raw الدائم
│   ├── manifest.py                   # قراءة وكتابة والتحقق من قوائم CSV للسلاسل
│   ├── progress.py                   # محرّك أشرطة التقدم الحيّة الآمن للخيوط (tqdm + بديل قياسي)
│   ├── runtime.py                    # كاشف البيئة الذكي (kaggle / full) وحارس الميزانية DiskBudgetGuard
│   ├── safe_zip.py                   # استخراج ذري بتمريرة واحدة مع فحص CRC-32 وحماية Zip-Slip
│   └── tcia_client.py                # عميل NBIA المتوازي عالي الإنتاجية مع البصمة اللحظية وتقارير السرعة
└── tests/                            # اختبارات الوحدة والبيانات الاصطناعية (24 اختبارًا)
```

وعند التشغيل، تتكون شجرة البيانات داخل `<data-root>` كالتالي:

```text
<data-root>/
├── raw/tcia/<case_id>/<role>/<series_uid>/   # ملفات DICOM المستخرجة كما وردت دون أي تعديل
└── manifests/
    ├── downloads/<series_uid>.json           # سجل كل سلسلة (الحالة، الحجم، السرعة، SHA-256، الاستخراج)
    ├── download_report.csv                   # جدول سرعات التنزيل والأحجام لكل سلسلة (MiB/s)
    ├── download_run_summary.json             # ملخص تشغيل التنزيل والسرعة المتوسطة واستهلاك حصة القرص
    ├── dicom_index.csv                       # فهرس تفصيلي لكل ملف DICOM على حدة
    ├── series_inventory.csv                  # جدول الفيزياء والتدقيق لكل سلسلة (Pitch، تغطية، حالة القبول)
    └── dicom_index_summary.json              # التقرير الشامل للفهرسة وزمن التنفيذ والربط بين السلاسل
```

---

## 4. دليل الأوامر الكامل (`hpct`)

### أ. فحص البيئة وحصة القرص والبطاقات الرسومية
```bash
hpct env
# أو لاختبار محاكاة وضع Kaggle محليًا:
hpct env --profile kaggle --data-root ./test_kaggle_data
```

### ب. اكتشاف سلاسل الصدر أو البطن تلقائيًا من TCIA وبناء ملف CSV
```bash
# جلب أول 12 حالة صدر (الإسقاطات كاملة الجرعة + الصور المنشورة فقط لتوفير 50% من المساحة):
hpct discover-tcia \
  --collection LDCT-and-Projection-data \
  --body-part CHEST \
  --roles projection_full,reconstruction_published \
  --max-cases 12 \
  --output configs/chest_pilot.csv
```

### ج. التحقق من ملف السلاسل قبل التنزيل
```bash
hpct validate-manifest --series-file configs/pilot_c012.csv
```

### د. التنزيل المتوازي الذكي مع شريط التقدم الحيّ (يتكيف تلقائيًا بين Kaggle $\le 18\text{ GiB}$ والخادم الكامل)
```bash
# نفس الأمر يعمل على Kaggle وعلى الخادم المتطور مع شريط تقدم حيّ وتقرير سرعة:
hpct download --series-file configs/pilot_c012.csv --workers 4 --retries 5
# وللتشغيل الصامت في السكربتات الخلفية بدون أشرطة تقدم:
hpct download --series-file configs/pilot_c012.csv --workers 4 --no-progress
```

### هـ. فهرسة الرؤوس المتوازية وتدقيق الهندسة والـ Pitch
```bash
# لفهرسة البيانات المنزّلة بسرعة بـ 8 خيوط متوازية وشريط تقدم حيّ:
hpct index --workers 8

# أو لفهرسة مجلد خارجي للقراءة فقط (مثل /kaggle/input أو قرص خارجي) دون نسخه:
hpct index --data-root data --scan-dir "/path/to/external/LDCT-and-Projection-data/C012" --workers 8
```

---

## 5. أوامر التشغيل والتحديث المباشرة على Kaggle

1. من إعدادات دفتر Kaggle:
   - فعّل **Internet = On**.
   - أضف توكن GitHub الخاص بك في **Add-ons $\rightarrow$ Secrets** تحت الاسم `GITHUB_TOKEN`.
   - في **Stage 0** يمكنك اختيار `Accelerator = None` لتوفير ساعات GPU، وعند الاختبارات الرسومية اللاحقة اختر **`GPU T4 x2`**.

2. شغّل الخلايا التالية في Kaggle (يدعم التحديث الفوري `git pull` إن كان المستودع مستنسخًا مسبقًا):

```python
import os
from pathlib import Path
from kaggle_secrets import UserSecretsClient

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
repo_dir = Path("/kaggle/working/HighPitchCT-Pipeline")
if not repo_dir.exists():
    os.system(f"git clone https://{token}@github.com/BrahimSoufghalem/HighPitchCT-Pipeline.git {repo_dir}")
else:
    os.system(f"git -C {repo_dir} pull --ff-only")

%cd /kaggle/working/HighPitchCT-Pipeline
!python -m pip install -q -e .
!python -m unittest discover -s tests -v
!hpct env
```

```bash
# تجربة التنزيل السريع وشريط التقدم الحيّ على عدة مرضى صدر (أو المريض C012):
!hpct download --series-file configs/pilot_c012.csv --workers 4 --retries 5
!hpct index --workers 8
```

---

## 6. بوابة قبول Stage 0 (شروط الانتقال إلى Stage 1)

- [x] الكشف الذكي عن بيئة `kaggle` مقابل `full`، وفرض سقف `18.0 GiB` وتفريغ `.zip` المؤقت في `/tmp` على Kaggle.
- [x] فكّ جميع الوسوم الخاصة `VR=UN` وفق `DICOM-CT-PD v4` وحساب الـ Pitch بالانحدار الخطي دون تحميل `PixelData`.
- [x] الربط الحتمي بين الصور والإسقاطات عبر `(0008,9121)` وكشف البتر المحوري للسلاسل غير المكتملة.
- [x] التحقق الفعلي على Kaggle للمريض الكامل `C012`: `351` شريحة صور (`350.00 mm`) + `13,446` ملف إسقاط (`1271.34 MB`، `FFSZ`، مدى `401.91 mm`، `pitch = 0.8966`، `RMSE = 0.02446 mm`، الحالة `VALID`).
- [x] ترقية سرعة التنزيل والفهرسة في `v0.3.1` (`requests.Session` مجمّع، كتل `4 MiB`، `SHA-256` أثناء التدفق، استخراج ذري بتمريرة واحدة، أشرطة تقدم `tqdm` حيّة، وتقرير `download_report.csv`، واجتياز `24/24` اختبارًا).

راجع الشرح التفصيلي المملّ لكل خوارزمية ومعادلة وبنية ملفات في مجلد [`docs/`](./docs/).
