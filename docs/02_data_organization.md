# Stage 0B — التوثيق التفصيلي لتنظيم وفهرسة `DICOM-CT-PD v4` وتدقيق الهندسة

يشرح هذا المستند بالتفصيل المملّ البنية الفيزيائية والبرمجية لملفات `DICOM-CT-PD` كما تحقّقنا منها فعليًا على المريض **`C012`** ودليل **`DICOM-CTPD User Manual Version 4`**، وكيف تعمل الوحدتان `ctpd_tags.py` و`index_dicom.py`.

---

## 1. مبدأ عدم المساس بالبيانات الخام (`Immutable Raw Data`)

- لا يُعدَّل أي ملف داخل `raw/tcia/` ولا يُعاد تسميته ولا تُغيَّر وسومه أبدًا.
- يدعم الأمر `hpct index` أيضًا فهرسة مجلدات خارجية للقراءة فقط عبر `--scan-dir` (مثل قرص خارجي أو `/kaggle/input/...`) مع كتابة جميع جداول الفهرسة والتقارير حصريًا في `<data-root>/manifests/`.
- لا يُحمَّل حقل `PixelData (7FE0,0010)` إطلاقًا أثناء الفهرسة (`stop_before_pixels=True`)، مما يتيح فهرسة أكثر من $13{,}000$ ملف إسقاط في ثوانٍ معدودة ودون استهلاك ذاكرة عشوائية.
- لا يُخزَّن `PatientID` نصيًا في ملفات الفهرسة، بل يُحفظ أول 12 محرفًا من تجزئة `SHA-256` (`patient_id_sha256_12`) لتجميع سلاسل المريض الواحد بأمان.

---

## 2. التصنيف الحتمي عبر `SOPClassUID`

لا يعتمد الكود على تخمين أسماء المجلدات، بل يصنّف كل ملف DICOM حتميًا عبر الوسم القياسي `(0008,0016) SOPClassUID`:

| قيمة `SOPClassUID` | التصنيف في `dicom_kind` | المعنى الفيزيائي |
| :--- | :--- | :--- |
| `1.2.840.10008.5.1.4.1.1.66` (مع وجود `(7029,1010)` و`(7031,1001)`) | `dicom_ct_pd_projection` | قراءة واحدة لمصفوفة الكاشف الكاملة عند زاوية وموضع بؤري واحد |
| `1.2.840.10008.5.1.4.1.1.2` | `ct_image` | شريحة CT معاد بناؤها تجاريًا (للمقارنة البصرية وفحص التغطية المحورية فقط) |
| غير ذلك | `other_dicom` / `raw_data_other` | ملفات أخرى غير مدعومة كإسقاطات مباشرة |

---

## 3. جدول فكّ الوسوم الخاصة (`VR = UN`) وفق الإصدار الرابع من الدليل

بسبب صيغة النقل `Implicit VR Little Endian (1.2.840.10008.1.2)` في بيانات TCIA، يقرأ `pydicom` جميع الوسوم الخاصة `(7029..7041)` بنوع `VR = UN` (مصفوفة بايتات ثنائية خام). تتولى الوحدة `ctpd_tags.py` فكّها بترتيب `Little-Endian` وفق الجدول التالي:

| الوسم `(Group,Elem)` | الاسم في الدليل الرسمي | النوع الفعلي (`VR`) | فكّ البايتات في Python | الوحدة والمعنى الفيزيائي (وقيمته في `C012`) |
| :--- | :--- | :---: | :---: | :--- |
| `(7029,1010)` | `NumberofDetectorRows` | `US` | `<H` | عدد صفوف الكاشف المحورية ($N_{\text{rows}} = 64$) |
| `(7029,1011)` | `NumberofDetectorColumns` | `US` | `<H` | عدد أعمدة الكاشف العرضية ($N_{\text{cols}} = 736$) |
| `(7029,1002)` | `DetectorElementTransverseSpacing` | `FL` | `<f` | عرض عنصر الكاشف عند سطح الكاشف ($d_u = 1.2858\text{ mm}$) |
| `(7029,1006)` | `DetectorElementAxialSpacing` | `FL` | `<f` | ارتفاع صف الكاشف عند سطح الكاشف ($d_v = 1.0947\text{ mm}$) |
| `(7029,100B)` | `DetectorShape` | `CS` | `ASCII` | شكل الكاشف (`CYLINDRICAL` / `SPHERICAL` / `FLAT`) |
| `(7031,1001)` | `DetectorFocalCenterAngularPosition` | `FL` | `<f` | الزاوية السمتية لمركز بؤرة الكاشف $\phi_0$ بالـ $\text{rad}$ |
| `(7031,1002)` | `DetectorFocalCenterAxialPosition` | `FL` | `<f` | الموضع المحوري لمركز بؤرة الكاشف $z_0$ بالـ $\text{mm}$ |
| `(7031,1003)` | `DetectorFocalCenterRadialDistance` | `FL` | `<f` | المسافة من مركز بؤرة الكاشف إلى مركز الدوران ($\rho_0 = 595.0\text{ mm}$) |
| `(7031,1031)` | `ConstantRadialDistance` | `FL` | `<f` | المسافة من مركز بؤرة الكاشف إلى سطح الكاشف ($d_0 = 1085.6\text{ mm}$) |
| `(7031,1033)` | `DetectorCentralElement` | `FL` ($\times 2$) | `<2f` | `[Column X, Row Y]` **بترقيم يبدأ من 1** (`[369.625, 32.5]`) |
| `(7033,100B)` | `SourceAngularPositionShift` | `FL` | `<f` | إزاحة البؤرة الزاوية $\Delta\phi$ بالـ $\text{rad}$ |
| `(7033,100C)` | `SourceAxialPositionShift` | `FL` | `<f` | إزاحة البؤرة المحورية $\Delta z$ بالـ $\text{mm}$ (`0` و`-0.66 mm`) |
| `(7033,100D)` | `SourceRadialDistanceShift` | `FL` | `<f` | إزاحة البؤرة الشعاعية $\Delta\rho$ بالـ $\text{mm}$ (`0` و`+5.45 mm`!) |
| `(7033,100E)` | `FlyingFocalSpotMode` | `CS` | `ASCII` | نمط البؤرة المتحركة (`FFSZ`، `FFSNONE`، `FFSXY`، `FFSXYZ`) |
| `(7033,1013)` | `NumberofSourceAngularSteps` | `US` | `<H` | عدد الإسقاطات في الدورة الكاملة (`1152`) |
| `(7033,1065)` | `PhotonStatistics` | `FL` ($\times N_{\text{cols}}$) | `<736f` | توزيع الفوتونات المكافئة عبر أعمدة الكاشف بعد مرشح Bowtie |
| `(7033,1067)` | `Timestamp` | `FL` | `<f` | الزمن المطلق للإسقاط بالـ $\text{ms}$ |
| `(7037,1009)` | `TypeofProjectionData` | `CS` | `ASCII` | `HELICAL` أو `AXIAL` |
| `(7039,1009)` | `LogFlag` | `CS` | `ASCII` | `YES` (تم تطبيق التحويل اللوغاريتمي مسبقًا؛ يُمنع تكراره) |
| `(7041,1001)` | `WaterAttenuationCoefficient` | `DS` | `float(ASCII)` | معامل توهين الماء للمعايرة ($\mu_{\text{water}} = 0.0192\text{ mm}^{-1}$) |

> **تنبيه حاسم للمرحلة القادمة (Stage 1):** في الرأس القياسي لملفات الإسقاط نجد `Rows = 736` و`Columns = 64` (بسبب حفظ المصفوفة من MATLAB بترتيب الأعمدة). لذلك عند قراءة البكسل في Python يجب أخذ المصفوفة المنقولة `ds.pixel_array.T` (أو `reshape((64, 736), order='F')`) للحصول على `[row, col] = (64, 736)`.

---

## 4. المعادلات الرياضية لتقدير الـ Pitch وتدقيق المسار أثناء الفهرسة

لكل سلسلة إسقاط خام، ترتب `index_dicom.py` القراءات حسب `InstanceNumber` ($k = 1, \dots, N_{\text{views}}$) وتنفّذ الخطوات التالية:

1. **فكّ التفاف الزوايا (`Angle Unwrapping`):**
   بما أن $\phi_0(k) \in [0, 2\pi)$ أو $[-\pi, \pi)$ تقفز عند إتمام كل دورة، نحسب المتتالية المتصلة $\phi_{\text{unwrap}}(k)$ بإزالة قفزات $2\pi$:
   $$\Delta\phi_k = \bigl((\phi_0(k) - \phi_0(k-1) + \pi) \bmod 2\pi\bigr) - \pi, \qquad \phi_{\text{unwrap}}(k) = \phi_{\text{unwrap}}(k-1) + \Delta\phi_k$$
   وإذا وُجدت أي خطوة تحقق $|\Delta\phi_k| > 3.5 \times \text{median}(|\Delta\phi|)$، تُصنَّف السلسلة فورًا بـ `INVALID_ANGLE_DISCONTINUITY`.

2. **حساب إزاحة الطاولة بالانحدار الخطي (`Linear Least-Squares Fit`):**
   نوائم العلاقة الخطية على كامل المسار:
   $$z_0(k) \approx a\,\phi_{\text{unwrap}}(k) + b$$
   حيث الميل $a$ بوحدة $\text{mm/rad}$. ومنه تكون إزاحة الطاولة الفعلية في الدورة الكاملة:
   $$F_{\text{rot}} = 2\pi\,|a| \quad [\text{mm/rotation}]$$
   ومعيار جودة انتظام حركة الطاولة هو الجذر التربيعي لمتوسط مربع البواقي:
   $$\text{RMSE}_z = \sqrt{\frac{1}{N_{\text{views}}} \sum_{k=1}^{N_{\text{views}}} \bigl(z_0(k) - (a\,\phi_{\text{unwrap}}(k) + b)\bigr)^2} \quad [\text{mm}]$$
   *(في المريض `C012` بلغ $\text{RMSE}_z = 0.026\text{ mm}$ فقط عبر `7,694` إسقاطًا!).*

3. **حساب التغطية الكلية عند مركز الدوران والـ Pitch اللابُعدي:**
   $$W_{\text{iso}} = N_{\text{rows}} \cdot d_v \cdot \frac{\rho_0}{d_0} \quad [\text{mm}], \qquad p_{\text{normalized}} = \frac{F_{\text{rot}}}{W_{\text{iso}}}$$
   *(في `C012`: $W_{\text{iso}} = 64 \times 1.09472 \times \frac{595.0}{1085.6} = 38.40\text{ mm}$، و$F_{\text{rot}} = 34.4317\text{ mm/rot}$، فيكون $p_{\text{normalized}} = 0.89666 \approx 0.9$).*

---

## 5. الربط الحتمي عبر `(0008,9121)` وكشف السلاسل المبتورة

أثبت الفحص المباشر على `C012` أن:
- `StudyInstanceUID` في الصور المنشورة يختلف عن `StudyInstanceUID` في الإسقاطات الخام!
- `FrameOfReferenceUID` يختلف أيضًا بين السلاسل الأربع كلها!

لذلك تستخرج `ctpd_tags.py` الرابط الرسمي من ترويسة شرائح الصور:
```text
(0008,9121) Reference Raw Data Sequence
 └── (0008,1115) Referenced Series Sequence
      └── (0020,000E) Series Instance UID  ---> UID سلسلة الإسقاطات الخام المطابقة تمامًا
```

ثم تقارن `index_dicom.py` المدى المحوري للإسقاطات $\Delta z_{\text{proj}} = z_{0,\max} - z_{0,\min}$ مع المدى المحوري للصور المرتبطة $\Delta z_{\text{img}}$:
- في الفحص الحلزوني المكتمل يجب أن يكون $\Delta z_{\text{proj}} \ge \Delta z_{\text{img}}$ (بسبب الـ *Over-ranging* عند طرفي المسح).
- فإذا كان $\Delta z_{\text{proj}} < 0.90\,\Delta z_{\text{img}}$ (أو كان عدد ملفات السلسلة أقل من سلسلة إسقاط شقيقة لنفس المريض)، تُوسم السلسلة بـ **`WARNING_TRUNCATED_PROJECTION_SERIES`** بدل قبولها خطأً لمجرد خلوّها من فجوات داخلية في `InstanceNumber`!

---

## 6. ملفات المخرجات الثلاثة للأمر `hpct index`

1. **`manifests/dicom_index.csv`:** سجل تفصيلي لكل ملف على حدة (`source_path`، `dicom_kind`، `instance_number`، `phi0_rad`، `z0_mm`، `slice_z_mm`، `referenced_raw_series_uid`، إلخ).
2. **`manifests/series_inventory.csv`:** الجدول العلمي الأهم لمراجعة السلاسل؛ يتضمن أبعاد الكاشف، نمط `FFS`، إزاحة الطاولة $F_{\text{rot}}$، الـ Pitch المحسوب $p_{\text{normalized}}$، خطأ المواءمة $\text{RMSE}_z$، السلاسل المرتبطة، وحالة القبول `validation_status`.
3. **`manifests/dicom_index_summary.json`:** ملخص مهيكل بجميع الإحصاءات وحالات القبول، جاهز للقراءة البرمجية في المراحل اللاحقة.
