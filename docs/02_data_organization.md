# Stage 0B — تنظيم وفهرسة DICOM

## سياسة عدم تعديل raw

المجلد `data/raw/tcia` نسخة مستخرجة كما وصلت. لا نعيد تسمية ملفات DICOM داخله، ولا نغيّر Tags، ولا نصحح الاتجاه. أي تحويل لاحق يذهب إلى `data/derived` مع provenance واضح.

## الفهرسة

الأمر `hpct index` يقرأ headers فقط باستخدام `stop_before_pixels=True`. هذا سريع ويمنع استهلاك الذاكرة عند ملفات الإسقاط الكبيرة. الحقول تشمل UIDات، modality، manufacturer، description، InstanceNumber، Rows، Columns، NumberOfFrames.

`PatientID` لا يُكتب مباشرة؛ يُحفظ أول 12 رمزًا من SHA-256 لتجميع الملفات دون نشر المعرّف النصي. هذا ليس anonymization جديدًا ولا بديلًا عن مراجعة سياسات البيانات.

## التصنيف

لا توجد في النسخة الحالية heuristic تلقائية تقول إن الملف projection أو image. الوسم الموجود في manifest تصريح أولي، ثم نتحقق من SOP Class والـ private tags وأبعاد PixelData في Stage 1. الهدف منع التصنيف الخاطئ الصامت.

## مخرجات القبول

- `dicom_index.csv`: صف لكل ملف مقروء أو فشل.
- `dicom_index_summary.json`: الأعداد الإجمالية، عدد الأخطاء، وعدد SeriesInstanceUID المختلفة.
- تقرير بشري لاحق يشرح كل خطأ بدل حذفه من النتائج.
