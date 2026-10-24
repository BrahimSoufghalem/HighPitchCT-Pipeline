# HighPitchCT-Pipeline

إطار بحثي وهندسي محكم وقابل للتدقيق لإعادة بناء وتقييم نموذج **ثنائي المجال (Dual-Domain Deep Learning)** لإعادة بناء التصوير المقطعي الحلزوني مرتفع الـ Pitch (**High-Pitch Helical CT**) انطلاقًا من بيانات الإسقاط الخام في مجموعة **TCIA (`LDCT-and-Projection-data`)**، ومُصمَّم خصيصًا لمذكرة ماستر في **الفيزياء الطبية** وفق نظام بوابات المراحل الصارم (**Stage-Gate**).

> **الإصدار الحالي:** `v0.3.0` — **المرحلة 0 (Stage 0) مكتملة ومُرقّاة**: اكتشاف السلاسل من واجهة TCIA، التنزيل الذكي الواعي ببيئة التشغيل (**سقف صارم $\le 18\text{ GiB}$ وتخزين مؤقت في `/tmp` تلقائيًا على Kaggle** مقابل تنزيل كامل غير مقيد على الحواسيب والخوادم)، التحقق من سلامة الأرشيف و`SHA-256`، وفهرسة متعمقة لرؤوس `DICOM-CT-PD v4` (فكّ ثنائي للوسوم الخاصة `VR=UN`، حساب ديناميكي للـ Pitch بالانحدار الخطي، الربط الحتمي بين الصور والإسقاطات عبر `(0008,9121)`، وكشف السلاسل المبتورة محوريًا).

---

## 1. جدول القرارات العلمية والهندسية المعتمدة (المرجع الدستوري للمشروع)

| المحور | القرار المعتمد | المبرر الفيزيائي والهندسي |
| :--- | :--- | :--- |
| **هوية البحث** | إعادة تنفيذ علمية مستقلة وموثّقة بالكامل مستلهمة من ملصق AAPM 2026 دون الاعتماد على مراسلة المؤلفين | ضمان الاستقلالية العلمية وتوثيق كل فرضية ومعادلة بدقة تصلح للمناقشة الأكاديمية |
| **منهجية العمل** | نظام **Stage-Gate** صارم: كود + توثيق تفصيلي مملّ + 4 مستويات اختبار (Unit / Synthetic / Integration / Real-data) لكل مرحلة | منع تراكم الأخطاء الهندسية وعدم الانتقال لأي مرحلة تالية قبل اجتياز بوابة القبول |
| **إدارة الـ Backends** | تحديد صريح للـ Backend في الإعدادات **ومنْع أي انتقال صامت (`No silent fallback`)** | ضمان قابلية تكرار التجارب وعدم خلط خوارزميات إعادة بناء مختلفة بين المرضى |
| **الوعي الذكي بالبيئة** | اكتشاف تلقائي للبيئة (`kaggle` مقابل `full`): على Kaggle يلتزم بسقف **$\le 18.0\text{ GiB}$** في `/kaggle/working` ويضع الـ `.zip` المؤقت في `/tmp` ويحذفه بعد الاستخراج؛ وخارج Kaggle ينزّل كامل القائمة | تمكين التجريب الآمن والسريع لـ $8\text{–}10$ مرضى على Kaggle ($2\times\text{T4}$)، ثم تشغيل الكود نفسه دون تعديل على خادم أكبر لكامل البيانات |
| **قراءة DICOM-CT-PD** | فكّ ثنائي صريح للوسوم الخاصة (`VR=UN`) وفق دليل `DICOM-CT-PD v4`، ترتيب وفحص استمرارية الزاوية و$z_0$، وتحويل المصفوفة بـ `pixel_array.T` (`order='F'`) إلى `(64, 736)` دون تكرار اللوغاريتم (`LogFlag=YES`) | ملفات TCIA مكتوبة بـ `Implicit VR Little Endian` وبترتيب أعمدة MATLAB (`Rows=736, Columns=64`)؛ أي قراءة بـ `order='C'` تخلط الصفوف بالأعمدة |
| **تقدير الـ Pitch والهندسة** | حساب ديناميكي لكل سلسلة عبر انحدار خطي لـ $z_0(\phi_{\text{unwrap}})$ لاستخراج `table_feed_mm_per_rot` و`normalized_pitch` واتجاه الطاولة | عدم افتراض ثبات الـ pitch عند `0.9` أو `0.6` أو ربطه بنوع العضو التشريحي |
| **البؤرة المتحركة (`FFS`)** | قراءة وتطبيق المركبات الثلاث $(\rho_0+\Delta\rho,\; \phi_0+\Delta\phi,\; z_0+\Delta z)$ لكل view | في جهاز `SOMATOM Definition AS+` حتى عند `FFSZ` تنزاح البؤرة شعاعيًا بـ $\Delta\rho = +5.45\text{ mm}$ بسبب ميل المصعد ($\approx 6.9^\circ$) |
| **الربط بين الصور والإسقاطات** | استخدام تسلسل `ReferenceRawDataSequence (0008,9121)` $\rightarrow$ `(0008,1115)` $\rightarrow$ `(0020,000E)` | في بيانات TCIA يختلف كلٌّ من `StudyInstanceUID` و`FrameOfReferenceUID` بين الصور المنشورة والإسقاطات الخام |
| **Curved-to-Flat & SSRB** | تحويل متجهي سريع مع `measured_mask` صريح وإلغاء `mode='nearest'` خارج الكاشف، واختيار القياس الأقرب لمركز الكاشف ($\arg\min_k \|v_k^\star\|$)، وفصل `Extrapolation` كمرحلة مستقلة | تمثيل نقص البيانات الفعلي الناتج عن الـ high-pitch بأمانة بدل إخفائه بتكرار أطراف الكاشف |
| **المرجع (`Reference Reconstruction`)** | إعادة بناء $x_{\text{reference}}$ من الإسقاطات الخام كاملة الجرعة عند الـ pitch الأصلي لكل فحص عبر خط `SSRB + Short-scan + Parker + FBP` الخاص بنا | منع الشبكة من تعلم تقليد مرشح إعادة البناء التجاري (`B50f` مثلًا) وإزاحات إحداثياته، وحصر التعلم في إزالة تشوهات الـ high-pitch |
| **محاكاة ITF وRDC والتعلم** | تحويل $x_{\text{reference}}$ إلى $\mu(x,y,z)$ وتوليد إسقاطات أمامية حلزونية (`LEAP`) لحالات ITF/RDC، وتدريب شبكتي Residual U-Net مرتبطتين عبر `Differentiable FBP` بخسارة $\mathcal{L}_1 + \mathcal{L}_\nabla$ | مطابقة الفيزياء الفعلية لمسار الطاولة وتطبيق البنية الرياضية الموصوفة في ملصق AAPM 2026 |

---

## 2. تفاصيل المرحلة الحالية: Stage 0 (`v0.3.0`)

تتكون المرحلة 0 من خمس وحدات برمجية متكاملة داخل `src/highpitchct/` دون تحميل أي مصفوفات بكسل في الذاكرة العشوائية:

1. **`runtime.py` — الكشف الذكي عن البيئة وحارس ميزانية القرص (`DiskBudgetGuard`):**
   - يكتشف تلقائيًا ما إذا كان الكود يعمل داخل **Kaggle** (عبر `KAGGLE_KERNEL_RUN_TYPE` أو وجود `/kaggle/working` و`/kaggle/input`) أم على جهاز/خادم عادي (`full`).
   - في وضع **`kaggle`**:
     - مسار البيانات الدائم: `/kaggle/working/highpitchct-data`.
     - مسار التنزيل المؤقت لملفات `.zip` و`.zip.part`: `/tmp/highpitchct-incoming` (خارج حصة الـ $20\text{ GB}$ الخاصة بـ `/kaggle/working` حتى لا يتضاعف استهلاك القرص أثناء فك الضغط!).
     - سقف الميزانية الصارم: **`18.0 GiB`** (`19,327,352,832` بايت). قبل تنزيل أي حالة (`case_id`) وقبل استخراج أي أرشيف ZIP، يتحقق `DiskBudgetGuard` من المساحة المستخدمة فعليًا + حجم الملفات غير المضغوطة؛ وإذا كان المريض التالي سيتجاوز سقف $18.0\text{ GiB}$، يتوقف التنزيل بأمان تام دون كسر البرنامج ويسجل `"status": "skipped_quota_reached"`، فتبقى جميع الحالات السابقة مكتملة وجاهزة للتدريب.
     - بعد التحقق من سلامة الـ ZIP وحساب بصمة `SHA-256` واستخراج الملفات، يُحذف ملف الـ `.zip` المؤقت من `/tmp` تلقائيًا، وعند إعادة تشغيل الأمر يتعرف الكود على الملفات المستخرجة (`already_extracted`) فلا يعيد تنزيلها أبدًا.
   - في وضع **`full`** (حاسوب محلي أو خادم متطور):
     - يُلغى سقف الـ $18\text{ GiB}$ (`max_disk_bytes = None`)، فينزّل الكود جميع السلاسل المفعّلة في ملف الـ CSV بالكامل وبنفس الأمر تمامًا.

2. **`manifest.py` و`tcia_client.py` — اكتشاف السلاسل والتنزيل الذري القابل للاستئناف:**
   - أمر `hpct discover-tcia`: يتصل بواجهة NBIA العامة، يجلب سلاسل مجموعة `LDCT-and-Projection-data`، يصنّف الأدوار تلقائيًا (`projection_full`، `reconstruction_published`، `projection_low`، `reconstruction_low`)، ويستخرج العدد المتوقع للملفات والحجم بالبايت لكل سلسلة لبناء ملف `CSV` جاهز وقابل للتكرار.
   - استئناف التنزيل عبر `HTTP Range` عندما يدعم الخادم `206 Partial Content`، والبدء الآمن من الصفر إذا أعاد `200 OK`.
   - فحص CRC كامل لكل عضو داخل الـ ZIP (`verify_zip`)، ومنع هجمات المسارات (`Zip Slip`) في `safe_extract_zip`.

3. **`ctpd_tags.py` و`index_dicom.py` — الفهرسة الفيزيائية لرؤوس `DICOM-CT-PD v4` وتدقيق السلاسل:**
   - قراءة الرؤوس فقط (`stop_before_pixels=True`) بسرعة عالية جدًا واستهلاك ذاكرة شبه معدوم.
   - دعم فهرسة المجلد المحلي `raw/tcia` أو أي مجلد خارجي للقراءة فقط عبر `--scan-dir` (مثل `/kaggle/input/...` على Kaggle أو قرص خارجي).
   - فكّ ثنائي دقيق للوسوم الخاصة ذات `VR=UN` وفق الجداول الرسمية لدليل `DICOM-CT-PD Version 4`.
   - حساب ديناميكي لكل سلسلة إسقاط خام:
     - فكّ التفاف الزوايا $\phi_{\text{unwrap}}$ وفحص القفزات الزاوية الشاذة.
     - مواءمة خطية $z_0 = a\,\phi_{\text{unwrap}} + b$ لحساب إزاحة الطاولة في الدورة $F = 2\pi|a|$، التغطية عند مركز الدوران $W_{\text{iso}} = N_{\text{rows}}\,d_v\,(\rho_0/d_0)$، والـ Pitch الفعلي اللابُعدي $p = F / W_{\text{iso}}$ مع خطأ المواءمة $\text{RMSE}$.
   - ربط الصور المنشورة بسلاسل الإسقاط الخام حتميًا عبر `ReferenceRawDataSequence (0008,9121)`، ومقارنة المدى المحوري $\Delta z_0$ للإسقاطات مع المدى المحوري $\Delta z_{\text{img}}$ للصور لكشف أي تنزيل مبتور جزئيًا (`WARNING_TRUNCATED_PROJECTION_SERIES`).

---

## 3. بنية المستودع والمخرجات

```text
HighPitchCT-Pipeline/
├── configs/
│   ├── pilot_c012.csv                # قائمة جاهزة للمريض C012 (فحص صدر كامل الجرعة + صوره المنشورة)
│   └── series.example.csv            # قالب توضيحي لملف السلاسل
├── docs/
│   ├── 00_project_overview.md        # النظرة العامة وخارطة المراحل التسع
│   ├── 01_data_download.md           # توثيق تفصيلي لآلية التنزيل وحارس الـ 18 GiB واكتشاف TCIA
│   ├── 02_data_organization.md       # توثيق تفصيلي لفك وسوم DICOM-CT-PD v4 وحساب الـ Pitch والتدقيق
│   ├── 03_kaggle.md                  # دليل تشغيل Kaggle (T4 x2 + 30GB RAM + سقف 18 GiB + Kaggle Secrets)
│   ├── validation_protocol.md        # بروتوكول التحقق وبوابات القبول الصارمة
│   └── decisions/                    # سجلات القرارات المعمارية (ADR-0001 إلى ADR-0004)
├── notebooks/
│   └── 00_kaggle_stage0.ipynb        # دفتر جاهز للتشغيل المباشر على Kaggle
├── src/highpitchct/
│   ├── cli.py                        # واجهة الأوامر الموحدة hpct
│   ├── ctpd_tags.py                  # مفكك الوسوم الثنائية الخاصة وفق DICOM-CT-PD v4 ورابط (0008,9121)
│   ├── index_dicom.py                # الفهرسة الفيزيائية وتدقيق اكتمال السلاسل وتقدير الـ Pitch
│   ├── layout.py                     # مسارات التخزين الحتمية وفصل incoming المؤقت عن raw الدائم
│   ├── manifest.py                   # قراءة وكتابة والتحقق من قوائم CSV للسلاسل
│   ├── runtime.py                    # كاشف البيئة الذكي (kaggle / full) وحارس الميزانية DiskBudgetGuard
│   ├── safe_zip.py                   # فحص CRC وحساب الحجم غير المضغوط والاستخراج الآمن
│   └── tcia_client.py                # عميل NBIA للتنزيل القابل للاستئناف واكتشاف المجموعات
└── tests/                            # اختبارات الوحدة والبيانات الاصطناعية (20 اختبارًا)
```

وعند التشغيل، تتكون شجرة البيانات داخل `<data-root>` كالتالي:

```text
<data-root>/
├── raw/tcia/<case_id>/<role>/<series_uid>/   # ملفات DICOM المستخرجة كما وردت دون أي تعديل
└── manifests/
    ├── downloads/<series_uid>.json           # سجل كل سلسلة (الحالة، الحجم، SHA-256، الاستخراج)
    ├── download_run_summary.json             # ملخص تشغيل التنزيل واستهلاك حصة القرص
    ├── dicom_index.csv                       # فهرس تفصيلي لكل ملف DICOM على حدة
    ├── series_inventory.csv                  # جدول الفيزياء والتدقيق لكل سلسلة (Pitch، تغطية، حالة القبول)
    └── dicom_index_summary.json              # التقرير الشامل للفهرسة والربط بين السلاسل
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

### د. التنزيل الذكي (يتكيف تلقائيًا بين Kaggle $\le 18\text{ GiB}$ والخادم الكامل)
```bash
# نفس الأمر يعمل على Kaggle وعلى الخادم المتطور دون تغيير:
hpct download --series-file configs/pilot_c012.csv --workers 2 --retries 5
```

### هـ. فهرسة الرؤوس وتدقيق الهندسة والـ Pitch
```bash
# لفهرسة البيانات المنزّلة في data_root الافتراضي:
hpct index

# أو لفهرسة مجلد خارجي للقراءة فقط (مثل /kaggle/input أو قرص خارجي) دون نسخه:
hpct index --data-root data --scan-dir "/path/to/external/LDCT-and-Projection-data/C012"
```

---

## 5. أوامر التشغيل المباشرة على Kaggle

1. من إعدادات دفتر Kaggle:
   - فعّل **Internet = On**.
   - أضف توكن GitHub الخاص بك في **Add-ons $\rightarrow$ Secrets** تحت الاسم `GITHUB_TOKEN` (لأن المستودع خاص).
   - في **Stage 0** يمكنك اختيار `Accelerator = None` لتوفير ساعات GPU، وعند الاختبارات الرسومية اللاحقة اختر **`GPU T4 x2`**.

2. شغّل الخلايا التالية في Kaggle:

```python
import os
from pathlib import Path
from kaggle_secrets import UserSecretsClient

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
repo_dir = Path("/kaggle/working/HighPitchCT-Pipeline")
if not repo_dir.exists():
    os.system(f"git clone https://{token}@github.com/BrahimSoufghalem/HighPitchCT-Pipeline.git {repo_dir}")

%cd /kaggle/working/HighPitchCT-Pipeline
!python -m pip install -q -e .
!python -m unittest discover -s tests -v
!hpct env
```

```bash
# تجربة المريض C012 الجاهز مباشرة (حجمه نحو 1.45 GB فقط، ضمن سقف 18 GiB):
!hpct validate-manifest --series-file configs/pilot_c012.csv
!hpct download --series-file configs/pilot_c012.csv --workers 2 --retries 5
!hpct index
```

---

## 6. بوابة قبول Stage 0 (شروط الانتقال إلى Stage 1)

- [x] الكشف الذكي عن بيئة `kaggle` مقابل `full`، وفرض سقف `18.0 GiB` وتفريغ `.zip` المؤقت في `/tmp` على Kaggle.
- [x] فكّ جميع الوسوم الخاصة `VR=UN` وفق `DICOM-CT-PD v4` وحساب الـ Pitch بالانحدار الخطي دون تحميل `PixelData`.
- [x] الربط الحتمي بين الصور والإسقاطات عبر `(0008,9121)` وكشف البتر المحوري للسلاسل غير المكتملة.
- [x] نجاح جميع اختبارات الوحدة والبيانات الاصطناعية + اختبار القبول الفعلي على بيانات المريض `C012`.
- [ ] تشغيل تجربة تنزيل وفهرسة واحدة على Kaggle للتأكد من سلاسة الشبكة ومخرجات `series_inventory.csv` قبل كتابة **Stage 1 (قارئ مصفوفات `DICOM-CT-PD` وبناء الفانتوم)**.

راجع الشرح التفصيلي المملّ لكل خوارزمية ومعادلة وبنية ملفات في مجلد [`docs/`](./docs/).
