# Stage 0A — التوثيق التفصيلي لتنزيل بيانات TCIA وإدارة حصة القرص الذكية

يشرح هذا المستند بالتفصيل المملّ كل قرار برمجي وهندسي داخل وحدات التنزيل (`manifest.py`، `layout.py`، `safe_zip.py`، `runtime.py`، `tcia_client.py`) في الإصدار `v0.3.0`.

---

## 1. الهدف من المرحلة 0A

توفير أداة سريعة، قابلة للتكرار العلمي (*Reproducible*)، وقابلة للاستئناف بعد انقطاع الشبكة، لتنزيل سلاسل الإسقاط الخام (`DICOM-CT-PD`) والصور المرجعية المنشورة من مجموعة `LDCT-and-Projection-data` عبر واجهة **NBIA REST API v1** العامة، مع **التكيف الذكي التلقائي** بين:
1. **بيئة Kaggle (`profile = kaggle`):** حيث مساحة القرص الدائمة في `/kaggle/working` محدودة بـ $\sim 20\text{ GB}$، فيطبّق الكود سقفًا صارمًا قدره **$18.0\text{ GiB}$** (`19,327,352,832` بايت) ويضع ملفات الـ `.zip` المؤقتة في `/tmp` خارج حصة `/kaggle/working` ثم ينظفها بعد التحقق والاستخراج.
2. **بيئة الحاسوب أو الخادم المتطور (`profile = full`):** حيث لا يوجد قيد الـ $20\text{ GB}$، فينزّل الكود كامل السلاسل المفعّلة في ملف الـ `CSV` دون أي سقف اصطناعي وبنفس الأوامر تمامًا.

---

## 2. واجهات NBIA REST API المستخدمة

تتعامل الوحدة `tcia_client.py` مع ثلاث نقاط نهاية رسمية عامة لا تتطلب مفتاح API للبيانات المفتوحة (`CHEST` و`ABDOMEN` تحت ترخيص CC BY 4.0):

1. **اكتشاف السلاسل (`getSeries`):**
   ```text
   GET https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeries?Collection=LDCT-and-Projection-data&BodyPartExamined=CHEST
   ```
   تُعيد قائمة JSON تتضمن لكل سلسلة: `PatientID`، `SeriesInstanceUID`، `SeriesDescription`، `BodyPartExamined`، `ImageCount` (عدد ملفات DICOM)، و`FileSize` (الحجم الإجمالي بالبايت).

2. **الاستعلام المسبق عن حجم السلسلة (`getSeriesMetaData`):**
   ```text
   GET https://services.cancerimagingarchive.net/nbia-api/services/v1/getSeriesMetaData?SeriesInstanceUID=<UID>
   ```
   يُستدعى تلقائيًا إذا لم يكن حقل `expected_bytes` مدونًا في ملف `series.csv` وكان حارس الميزانية (`DiskBudgetGuard`) مفعّلًا، لمعرفة حجم السلسلة بالبايت **قبل** بدء تنزيلها.

3. **تنزيل أرشيف السلسلة (`getImage`):**
   ```text
   GET https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage?SeriesInstanceUID=<UID>
   ```
   يُعيد دفقًا ثنائيًا لملف `ZIP` يحتوي جميع ملفات `.dcm` الخاصة بتلك السلسلة.

---

## 3. لماذا نستبعد سلاسل الجرعة المنخفضة افتراضيًا؟ (حساب البايتات الفعلي من المريض `C012`)

عند فحص المريض الحقيقي **`C012`** (فحص صدر كامل على جهاز `SIEMENS SOMATOM Definition AS+`) وجدنا أن لكل مريض في TCIA أربع سلاسل:
1. `Full-dose-projections` $\leftarrow$ الدور: `projection_full` (حجم الملف الواحد `99,144` بايت $\approx 96.8\text{ KB}$؛ فحص الصدر الكامل $\sim 13{,}000$ ملف $\approx \mathbf{1.25\text{–}1.35\text{ GB}}$).
2. `Full Dose Images` $\leftarrow$ الدور: `reconstruction_published` (`351` شريحة، الحجم الإجمالي `184.84 MB` $\approx \mathbf{0.18\text{ GB}}$، تُستعمل للتحقق البصري ومراجعة التغطية المحورية فقط، **وليست Ground Truth للتدريب**).
3. `Low dose projections` $\leftarrow$ الدور: `projection_low` ($\approx 1.25\text{–}1.35\text{ GB}$).
4. `Low Dose Images` $\leftarrow$ الدور: `reconstruction_low` ($\approx 0.18\text{ GB}$).

بما أن موضوع مذكرتنا يدرس **التصوير مرتفع الـ Pitch (`High-Pitch Helical CT`)** انطلاقًا من المرجع كامل الجرعة، فلا حاجة لتنزيل `projection_low` و`reconstruction_low` في التجارب الأساسية. استبعادهما يوفّر **50% من مساحة القرص وزمن التنزيل فورًا**، فيصبح حجم المريض الكامل **`~1.45 GB` فقط**، مما يسمح بوضع **$8$ إلى $10$ مرضى كاملين** داخل سقف الـ **$18.0\text{ GiB}$** على Kaggle!

---

## 4. بنية ملف الإدخال `series.csv`

الأعمدة الإلزامية الخمسة (مع دعم ثلاثة أعمدة اختيارية إضافية يملؤها `hpct discover-tcia` تلقائيًا):

| العمود | إلزامي؟ | الوصف والقيود الصارمة |
| :--- | :---: | :--- |
| `case_id` | نعم | معرّف الحالة الداخلي مثل `case_C012` (يُطهر تلقائيًا من أي محارف غير آمنة في المسارات) |
| `series_instance_uid` | نعم | معرّف DICOM الرقمي المنقّط فقط (`^[0-9]+(\.[0-9]+)+$`، الطول $\le 64$)؛ يُرفض التكرار صراحةً |
| `role` | نعم | أحد الأدوار الخمسة فقط: `projection_full`، `projection_low`، `reconstruction_published`، `reconstruction_low`، `unknown` |
| `body_part` | اختياري | مثل `CHEST` أو `ABDOMEN` |
| `expected_files` | اختياري | العدد المتوقع لملفات DICOM في السلسلة؛ إن وُجد يتحقق الكود بعد الاستخراج من مطابقته تمامًا |
| `expected_bytes` | اختياري | الحجم المتوقع بالبايت؛ يُستخدم لحساب ميزانية القرص مسبقًا قبل التنزيل |
| `description` | نعم | وصف نصي حر للسلسلة |
| `enabled` | نعم | قيمة منطقية (`true` / `false`) لتفعيل أو تعطيل السلسلة دون حذف سطرها |

---

## 5. الخوارزمية التفصيلية للتنزيل وحراسة ميزانية الـ $18.0\text{ GiB}$

عند استدعاء `hpct download`:

1. **تحديد ملف البيئة (`resolve_environment_profile`):**
   - إذا كان `--profile auto` (الافتراضي): يفحص وجود متغير `KAGGLE_KERNEL_RUN_TYPE` أو المجلدين `/kaggle/working` و`/kaggle/input`.
   - في وضع `kaggle`:
     - `data_root = /kaggle/working/highpitchct-data`
     - `incoming_root = /tmp/highpitchct-incoming`
     - `max_disk_bytes = 18 * 1024^3 = 19,327,352,832` بايت
     - `cleanup_archive_after_extract = True`
   - في وضع `full`:
     - `data_root = ./data` (أو ما يحدده `--data-root`)
     - `incoming_root = <data_root>/incoming`
     - `max_disk_bytes = None` (بلا سقف)
     - `cleanup_archive_after_extract = False`

2. **التجميع على مستوى المريض (`Case-Level Grouping`):**
   - تُجمَّع السلاسل المفعّلة حسب `case_id` بنفس ترتيب ظهورها في `series.csv`.
   - الهدف العلمي: منع تنزيل `projection_full` لمريض ما ثم التوقف بسبب الحصة قبل تنزيل `reconstruction_published` المرافقة له (أو ترك نصف حالة).

3. **الفحص المسبق للموجود على القرص (`Idempotent Skip`):**
   - إذا كانت السلسلة مستخرجة مسبقًا داخل `raw/tcia/<case_id>/<role>/<uid>/` وحالتها في `manifests/downloads/<uid>.json` مكتملة وعدد ملفاتها مطابق، تُوسم فورًا بـ `"already_extracted"` دون أي اتصال بالشبكة — **حتى لو كان ملف الـ `.zip` الأصلي قد حُذف من `/tmp` على Kaggle**.

4. **الحجز المسبق والرقابة المزدوجة على الحجم (`Two-Phase Quota Guard`):**
   - **المرحلة الأولى (قبل التنزيل):** يحسب الكود مجموع `expected_bytes` للسلاسل المتبقية من الـ `case_id` الحالي. إذا كان:
     $$\text{Bytes}_{\text{on\_disk}}(\texttt{data\_root}) + \text{Bytes}_{\text{case}} > \text{Bytes}_{\max}$$
     يمتنع الكود عن بدء تنزيل هذا المريض وجميع المرضى اللاحقين، ويكتب سجلًا صريحًا `"status": "skipped_quota_reached"` مع تفاصيل البايتات المتبقية.
   - **المرحلة الثانية (قبل فك الضغط مباشرةً في `safe_extract_zip`):** حتى لو لم تتوفر بيانات الحجم المسبقة من الخادم، ينزل ملف `.zip` داخل `/tmp` أولًا، ثم يقرأ الكود جدول محتويات الأرشيف (`archive.infolist()`) ويجمع الحجم الفعلي غير المضغوط لجميع الملفات:
     $$\text{Bytes}_{\text{uncompressed}} = \sum_{m \in \text{archive}} m.\texttt{file\_size}$$
     فإن كان استخراجها سيتجاوز `max_disk_bytes` في `data_root`، يُرفع الاستثناء `DiskBudgetExceededError` قبل كتابة بايت واحد في `data_root`، ويُنظَّف الأرشيف المؤقت بأمان.

5. **استئناف التنزيل (`HTTP Range`) وسلامة الأرشيف:**
   - يكتب الدفق دائمًا في `<uid>.zip.part`.
   - إذا وُجد جزء سابق، يرسل الترويسة `Range: bytes=<existing>-`.
   - إذا استجاب الخادم بـ `HTTP 206 Partial Content` يُفتح الملف بوضع الإلحاق `"ab"`؛ أما إذا استجاب بـ `HTTP 200 OK` فيُعاد فتحه من الصفر `"wb"` لمنع فساد ملف الـ ZIP.
   - يُفحص الأرشيف كاملًا بـ `zipfile.ZipFile.testzip()` قبل تحويل اسمه ذريًا (`os.replace`) إلى `<uid>.zip`.
   - تُحسب بصمة `SHA-256` وتُحفظ في `manifests/downloads/<uid>.json`، ثم يُحذف `<uid>.zip` إن كان `cleanup_archive_after_extract = True`.

---

## 6. تشخيص الأخطاء المحتملة وحلولها

| الرسالة أو الحالة | السبب الدقيق | الإجراء الصحيح |
| :--- | :--- | :--- |
| `skipped_quota_reached` | مجموع البيانات وصل إلى سقف الـ $18.0\text{ GiB}$ المخصص لـ Kaggle | سلوك طبيعي ومقصود على Kaggle؛ انتقل مباشرةً إلى `hpct index` على الحالات المكتملة |
| `Extracted DICOM count (X) does not match expected_files (Y)` | الخادم أعاد أرشيفًا ناقصًا مقارنةً بوصف السلسلة | احذف مجلد السلسلة المستخرج وأعد تشغيل `hpct download` |
| `Corrupt ZIP member` | تلف في البايتات أثناء النقل الشبكي | يعيد الكود المحاولة تلقائيًا حتى `--retries` بالتأخير الأسي (`Exponential Backoff`) |
| `/kaggle/input is read-only` | محاولة تمرير `--data-root /kaggle/input/...` | استخدم `/kaggle/working/highpitchct-data` للإخراج، ومرر `/kaggle/input/...` عبر `--scan-dir` فقط عند الفهرسة |
