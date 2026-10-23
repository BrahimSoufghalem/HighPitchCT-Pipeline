# HighPitchCT-Pipeline

مسار بحثي قابل للتدقيق لإعادة بناء فكرة **high-pitch helical CT** انطلاقًا من بيانات الإسقاط الخام في TCIA. الهدف ليس نسخ نتائج الملصق حرفيًا، بل بناء نسخة علمية واضحة يمكن الدفاع عنها في مذكرة ماستر في الفيزياء الطبية.

> **الحالة الحالية:** Stage 0 فقط — تنزيل البيانات، التحقق منها، فكّها بأمان، وفهرسة رؤوس DICOM. لا يحتوي هذا الإصدار بعد على SSRB أو FBP أو ITF/RDC أو شبكة عصبية.

## المبادئ غير القابلة للتفاوض

1. **لا افتراض ثابت للـ pitch.** سيُقدَّر لاحقًا لكل Series من مسار الطاولة والزوايا؛ لا نربطه تلقائيًا بالصدر أو البطن.
2. **لا Ground Truth مجهول.** الصور المنشورة من الجهاز تُستخدم للمقارنة البصرية فقط، بينما مرجع التدريب سيُعاد بناؤه من الإسقاطات كاملة الجرعة وبـ pitch الأصلي عبر خطنا نفسه.
3. **لا تعويض صامت.** أي عينة غير مقاسة بعد SSRB ستحمل mask صريحًا؛ لن نستخدم nearest extrapolation لإخفائها.
4. **Stage-Gate.** لا نبدأ مرحلة جديدة قبل توثيق واختبار وقبول المرحلة السابقة على بيانات اصطناعية وبيانات TCIA حقيقية.
5. **قابلية التكرار.** كل تنزيل له Manifest وحجم وSHA-256 وحالة محفوظة.

## ما الذي ينفذه الإصدار 0.2.0؟

- قراءة ملف CSV يحدد SeriesInstanceUID المطلوب بدقة.
- تنزيل ZIP لكل سلسلة عبر واجهة NBIA العامة.
- تنزيل متوازي على مستوى السلاسل، مع retries وexponential backoff.
- استكمال ملف `.part` عندما يرد الخادم بـ HTTP 206؛ وإعادة آمنة من الصفر إذا تجاهل Range.
- فحص سلامة ZIP قبل اعتماده، ثم rename ذري.
- منع Zip Slip أثناء الاستخراج.
- إنشاء سجل JSON لكل تنزيل يتضمن SHA-256.
- فهرسة DICOM دون تحميل PixelData، مع تجزئة PatientID بدل تخزينه نصيًا.
- اكتشاف Kaggle واختيار `/kaggle/working/highpitchct-data` تلقائيًا.
- أمر `hpct env` لفحص RAM والقرص وبطاقات GPU قبل بدء أي تجربة.

## التجهيز المحلي

```bash
bash scripts/bootstrap.sh
source .venv/bin/activate
```

أو يدويًا:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
```

## التشغيل على Kaggle

يوجد Notebook جاهز في `notebooks/00_kaggle_stage0.ipynb`. في Kaggle:

1. فعّل **Internet** لتنزيل المستودع وبيانات TCIA.
2. Stage 0 لا يحتاج GPU؛ استخدم CPU كي لا تستهلك حصة GPU.
3. عند مراحل reconstruction/training اختر **GPU T4 x2**.
4. لا تكتب داخل `/kaggle/input`؛ الأداة تختار تلقائيًا `/kaggle/working/highpitchct-data`.

```bash
!git clone https://github.com/BrahimSoufghalem/HighPitchCT-Pipeline.git
%cd HighPitchCT-Pipeline
!python -m pip install -q -e .
!hpct env
```

التفاصيل وحدود الذاكرة والقرص موجودة في `docs/03_kaggle.md`.

## تجهيز قائمة البيانات

انسخ المثال:

```bash
cp configs/series.example.csv configs/series.csv
```

ثم ضع سطرًا لكل سلسلة:

```csv
case_id,series_instance_uid,role,description,enabled
pilot_001,1.2.840....,projection_full,Full-dose raw projection series,true
pilot_001,1.2.840....,reconstruction_published,Vendor image series for visual comparison,true
```

القيم المسموحة في `role`:

- `projection_full`: الإسقاطات كاملة الجرعة المرشحة لصنع Reference Reconstruction.
- `projection_low`: إسقاطات منخفضة الجرعة إذا وُجدت.
- `reconstruction_published`: صور الجهاز المنشورة؛ ليست Ground Truth للتدريب.
- `unknown`: عندما لم نتحقق بعد من نوع السلسلة.

تحقق من الملف دون تنزيل:

```bash
hpct validate-manifest --series-file configs/series.csv
```

## التنزيل والفهرسة

على الحاسوب المحلي يمكن تحديد المسار صراحةً:

```bash
hpct download --series-file configs/series.csv --data-root data --workers 2 --retries 5
hpct index --data-root data
```

على Kaggle يكفي حذف `--data-root`:

```bash
hpct download --series-file /kaggle/working/series.csv --workers 2 --retries 5
hpct index
```

البنية الناتجة:

```text
<data-root>/
├── incoming/                 # ZIP النهائي وملفات .part المؤقتة
├── raw/tcia/<case>/<role>/   # DICOM المستخرج دون تعديل
└── manifests/
    ├── downloads/*.json      # حالة، حجم، hash، مسار الاستخراج
    ├── dicom_index.csv       # فهرس رؤوس DICOM
    └── dicom_index_summary.json
```

## بوابة قبول Stage 0

لا نبدأ قارئ الإسقاطات والهندسة قبل تحقق الآتي على Pilot حقيقي:

- [ ] كل Series مطلوبة نزلت دون ملف ZIP تالف.
- [ ] إعادة الأمر لا تعيد تنزيل السلاسل المكتملة.
- [ ] انقطاع تنزيل اختباري يترك `.part` ويمكن استكماله أو إعادة تشغيله بأمان.
- [ ] `dicom_index.csv` لا يحتوي أخطاء قراءة غير مفسرة.
- [ ] عدد الملفات والسلاسل منطقي مقارنة بصفحة TCIA/NBIA.
- [ ] عينة الإسقاط المرفقة تظهر داخل فئة projection بعد التحقق اليدوي من Tags الخاصة بـ Siemens.
- [ ] تقرير قبول قصير محفوظ تحت `reports/stage_0/`.

## خريطة الطريق

0. **البيانات والفهرسة** — الإصدار الحالي.
1. قارئ DICOM-CT-PD واتجاه المصفوفة، مع اختبار phantom صغير.
2. استخراج الهندسة وتقدير pitch لكل سلسلة مع الاحتفاظ بإشارة اتجاه الطاولة وFFS.
3. Curved-to-flat سريع ومتحقق عدديًا.
4. SSRB مع measured/missing mask واختبارات هندسية.
5. short-scan selection + Parker weighting + extrapolation منفصل.
6. FBP مرجعي موحد لإنتاج Reference Reconstruction.
7. محاكاة ITF وRDC مع بروتوكول واضح لمنع data leakage.
8. بناء الأزواج التدريبية ثم النموذج ثنائي المجال.

راجع التوثيق التفصيلي في `docs/` قبل التشغيل.
