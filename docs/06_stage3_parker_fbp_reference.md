# التوثيق الرياضي والفيزيائي والبرمجي التفصيلي للمرحلة الثالثة (`Stage 3 — v0.6.0 / v0.7.0`): استكمال الجوانب المفقودة (`Sinogram Completion`)، ترجيح `Parker` الدقيق للمسح القصير، إعادة البناء المرجعية `Ram-Lak Fan-Beam FBP` ($x_{\text{reference}}$)، ومعايرة وحدات هاونسفيلد (`0.0 HU`)

> **تحديث `v0.7.0`:** تم اعتماد مرشح **`ram_lak` (`Ram-Lak / Ramp Filter`)** مرشحًا رئيسيًا افتراضيًا بدلًا من `hann` للحفاظ على كامل الدقة المكانية العالية، مع مطابقة مجال الرؤية الفيزيائي (`FOV = 400.0 mm`) تلقائيًا مع صور الشركة المنشورة (`Published Vendor CT`) لضمان تطابق مقياس الرسم المليمتري `1:1`.

تُمثّل **المرحلة الثالثة (`Stage 3`)** الحلقة الفيزيائية الحاسمة التي تحوّل سينوغرامات المروحة المستوية ثنائية الأبعاد (`2D Planar Fan-Beam Sinograms`) الناتجة من `Stage 2` (`Curved-to-Flat + Helical SSRB`) إلى **مقاطع تصوير مقطعي كمّية مُعايرة بوحدات هاونسفيلد (`Hounsfield Units, HU`)** ومعامل التوهين الخطي $\mu(x, y, z)\;[\text{mm}^{-1}]$، وتُولّد المرجع الفيزيائي الموحّد **$x_{\text{reference}}$** الذي ستُبنى عليه محاكاة الـ `High-Pitch` (`Stage 4`) وتدريب وتقييم الشبكة ثنائية المجال (`Stages 6–8`).

---

## 1. لماذا نبني $x_{\text{reference}}$ بخط `FBP` الخاص بنا ولا نستخدم صور الشركة المنشورة (`Vendor CT`) هدفًا للتدريب؟

أثناء الفحص الفعلي لبيانات المريض الحقيقي `C012` على `Kaggle`، تبيّنت ثلاث حقائق هندسية وفيزيائية قاطعة في بيانات `TCIA (LDCT-and-Projection-data)`:

| البُعد الفيزيائي | مساقط `DICOM-CT-PD` الخام (ومرجعنا $x_{\text{reference}}$) | صور `CT Image Storage` المنشورة من جهاز `Siemens` | ماذا لو درّبنا الشبكة على صور `Siemens` مباشرةً؟ |
| :--- | :--- | :--- | :--- |
| **مبدأ جملة الإحداثيات (`Origin`)** | محور دوران الجانتري (`Isocenter`): $(x, y) = (0, 0)$ عند مركز الدوران تمامًا | جملة `LPS` مرتبطة بالطاولة بـ `FrameOfReferenceUID` مختلف تمامًا (في `C012`: `IPP = [-199.61, -368.11, z]`، أي إزاحة رأسية اصطناعية عن مركز الجانتري!) | ستُجبر الشبكة العصبية على تعلّم إزاحة مكانية واستيفاء هندسي بدل تعلّم فيزياء نقص البيانات! |
| **مرشح إعادة البناء (`Kernel`)** | مرشح تحليلي معروف ومثبت رياضيًا (`Spatial Ram-Lak + Hann` قابل للاشتقاق تمامًا في `Stage 6`) | مرشح تجاري مغلق حاد الحواف (`B50f` للرئة مع تعزيز ترددات عالية غير خطي وتصحيحات تجارية للتشتت وتصلب الحزمة) | ستتعلّم الشبكة تقليد رنين مرشح `B50f` التجاري (`Kernel Style Transfer`) بدل إزالة تشوهات الـ `High-Pitch` |
| **معامل التوهين المرجعي للماء** | مرتبط صراحةً بوسم `WaterAttenuationCoefficient` في رأس المسقط ($\mu_{\text{water}} \approx 0.019257\;\text{mm}^{-1}$ في `C012`) | معاد المعايرة داخليًا بعد جداول تصحيح خاصة بالشركة | انحياز كمّي (`HU Bias`) بين الإسقاط الأمامي في `Stage 4` والصورة الهدف |

لهذا السبب يعتمد مشروعنا (وفق سجل القرار المعماري [`ADR-0001`](./decisions/0001-reference-reconstruction.md)) بناء **$x_{\text{reference}}$** من مساقط الجرعة الكاملة عند الـ `Pitch` الأصلي المنخفض نسبيًا ($\approx 0.90$ في `C012` حيث نسبة القياس في `Short-Scan` تبلغ **`100.00%`** دون أي شعاع مفقود!) باستخدام المحرّك الفيزيائي الدقيق نفسه (`SSRB + Exact Parker + Spatial Ram-Lak/Hann FBP`)، بينما تُحفظ صور `Siemens` المنشورة جنبًا إلى جنب في لوحات المقارنة البصرية (`Visual QA`) مرجعًا تشريحيًا خارجيًا مستقلًا.

### اتفاقية التوجيه الإشعاعي السريري (`Supine Radiological Pixel Grid`)
في جملة الإحداثيات اليسرى (`Left-Handed Coordinate System`) للشكل 5 من دليل `DICOM-CT-PD User Manual v4` (المرسومة من جهة قاعدة الطاولة باتجاه الجانتري $+z$):
- المحور $+x$ يتجه إلى **يمين المريض (`Patient Right` — الكبد والرئة اليمنى)**، والمحور $-x$ إلى **يسار المريض (`Patient Left` — القلب والرئة اليسرى)**.
- المحور $+y$ يتجه إلى الأعلى (**الساعة 12 — عظم القص الأمامي `Anterior Sternum`**)، والمحور $-y$ إلى الأسفل (**الساعة 6 — العمود الفقري الخلفي `Posterior Spine`**).

ولكي تظهر جميع مقاطعنا المقطعية بالتوجيه الطبي الإشعاعي القياسي العالمي (`Radiological View`: النظر من قدمي المريض نحو رأسه، بحيث يقع عظم القص في أعلى الصورة `Row 0`، العمود الفقري في أسفلها `Row N_xy-1`، يمين المريض/الكبد على يسار المشاهد `Col 0`، ويسار المريض/القلب على يمين المشاهد `Col N_xy-1`، مطابقًا تمامًا لصور `DICOM CT Image Storage` المنشورة)، تُعرّف [`reconstruct_fanbeam_fbp`](../src/highpitchct/recon_fbp.py) شبكة البكسلات `[iy, ix]` بحجم $N_{xy} \times N_{xy}$ وعرض ساحة رؤية $\text{FOV}$ كالتالي:

$$d_{xy} = \frac{\text{FOV}}{N_{xy}}, \qquad x_{\text{ix}} = x_c + \left(\frac{N_{xy} - 1}{2} - \text{ix}\right) d_{xy} \quad (\text{Col } 0 = +x\text{ [Patient Right]} \rightarrow \text{Col } N_{xy}-1 = -x\text{ [Patient Left]})$$

$$y_{\text{iy}} = y_c + \left(\frac{N_{xy} - 1}{2} - \text{iy}\right) d_{xy} \quad (\text{Row } 0 = +y\text{ [Anterior]} \rightarrow \text{Row } N_{xy}-1 = -y\text{ [Posterior]})$$

مع قناع دائري صريح لساحة الرؤية الفيزيائية للجهاز (`Scanner FOV`):

$$R_{\text{FOV}} = \rho_0 \sin\gamma_{\max}, \qquad \text{fov\_mask}[\text{iy}, \text{ix}] = \bigl((x_{\text{ix}} - x_c)^2 + (y_{\text{iy}} - y_c)^2 \le R_{\text{FOV}}^2\bigr)$$

خارج هذه الدائرة، يُضبط $\mu = 0\;\text{mm}^{-1}$ و$\text{HU} = -1000\;\text{HU}$ صراحةً.

---

## 2. استكمال السينوغرام الفيزيائي ومعالجة بتر الأطراف (`extrapolate_sinogram`)

### 2.1 قدسية قناع القياس الأصلي (`Immutability of measured_mask`)
يُعيد `Stage 2` كائن [`SSRBResult`](../src/highpitchct/rebinning.py) متضمّنًا `sinograms` (حيث كل نقطة غير مقاسة تساوي `0.0`) والقناع المنطقي الصارم `measured_mask`. في المرحلة السابعة (`Stage 7`)، ستحتاج شبكة `Sinogram Residual U-Net` إلى هذا القناع الأصلي ذاته لتعرف بدقة أين تقع الفجوات الفيزيائية الحقيقية. لذلك تلتزم الدالة [`extrapolate_sinogram`](../src/highpitchct/recon_fbp.py) بقاعدة صارمة مفحوصة آليًا في الاختبارات: **لا يُعدَّل `SSRBResult.measured_mask` إطلاقًا**، بل يُنسخ إلى `PreparedSinogram.original_measured_mask` ويُبنى فوقه سينوغرام مكتمل `completed_sinograms` لأغراض `FBP`.

### 2.2 لماذا لا يكفي وضع الأصفار مباشرةً في `Parker + FBP` عند الـ `High-Pitch`؟
ترجيح `Parker` يفترض أن الشعاع $(\tilde{\beta}, \alpha_m)$ ومرافقه الهندسي $(\tilde{\beta} + 2\alpha_m \pm \pi, -\alpha_m)$ يحملان التكامل الخطي نفسه $g$، فيضرب الأول في $w_1$ والثاني في $w_2$ حيث $w_1 + w_2 = 1$. إذا سقط الشعاع الأول في فجوة `High-Pitch` وبقي صفرًا ($g_1 = 0$) بينما قيس مرافقه ($g_2 = g$)، فإن المجموع المرجّح يصبح:

$$w_1 \cdot 0 + w_2 \cdot g = w_2 g < g$$

أي هبوط كاذب في التوهين، يتلوه قفز حاد عند حافة الفجوة يحوّله مرشح `Ram-Lak` التفاضلي إلى **خطوط تشوه شعاعية عنيفة (`Severe Missing-Wedge Streaks`)** تخترق كامل الرئة والقلب!

### 2.3 الاستكمال الفيزيائي بالأشعة المرافقة متبوعًا بالدمج الاتجاهي (`inpaint_mode="conjugate_feather"`)
تنفّذ الدالة [`extrapolate_sinogram`](../src/highpitchct/recon_fbp.py) أربع خطوات متتالية في وضع `conjugate_feather` الافتراضي:

0. **التمهيد المسبق للأعمدة الحدية ذات إزاحة ربع عنصر الكاشف (`Quarter-Offset Border Column Pre-fill`):**
   تُملأ أي أعمدة طرفية غير مقاسة بالكامل من أقرب عمود جارٍ مقاس عند الزوايا المقاسة نفسها قبل البحث عن الأشعة المرافقة.
1. **الإنقاذ الفيزيائي الدقيق من الشعاع المرافق المقاس (`Exact Planar Conjugate-Ray Fill`):**
   لكل بكسل مفقود $(\beta_k, u_m)$ في إحدى منطقتي التداخل الزاوي (`Overscan Zones`)، نحسب إحداثيات شعاعه المرافق المستوي تمامًا (بما أن زاوية الشعاع الكلية هي $\theta = \beta_k + \alpha_m$ وعند عكس الاتجاه بـ $\pm\pi$ يصبح $\alpha' = -\alpha_m$ فإن $\beta' - \alpha_m = \beta_k + \alpha_m \pm \pi$، أي أن إشارة $+2\alpha_m$ موجبة دائمًا لكلا الاتجاهين $\pm\pi$!):
   $$\beta_{\text{conj}} = \beta_k + 2\alpha_m \pm \pi, \qquad u_{\text{conj}} = -u_m \quad \bigl(\alpha_m = \arctan(u_m / d_0)\bigr)$$
   ونتحقق بشرط صارم: **لا يُقبل الشعاع المرافق إلا إذا وقع داخل شبكة الزوايا والأعمدة وكانت جميع الجيران الأربعة للاستيفاء الثنائي مقاسةً فعليًا (`True`)!** عندها يُملأ البكسل المفقود بقيمة قياس حقيقي لنفس المستقيم الأفقي عبر جسم المريض، ويُسجَّل في `conjugate_filled_mask`.
2. **الدمج السلس للزوايا والأعمدة المتبقية (`Directional Angular + Transverse Feathering`):**
   للفجوات المتبقية التي غاب فيها الشعاع ومرافقه معًا (عند `pitch` مرتفع جدًا)، يُطبَّق استيفاء خطي عبر الزاوية $\beta$ متبوعًا بتنعيم ثنائي البعد `(k, u)` داخل المنطقة غير المقاسة أصلاً (`~mask_orig`) حصريًا (مع بقاء جميع القياسات الأصلية `mask_orig` محمية بنسبة `100%`) لمنع أي انقطاع قطري (`Diagonal Step Discontinuity`) قبل الفلترة بمرشح `Ram-Lak`.
3. **التمديد الجانبي بجيب التمام المرفوع (`Raised-Cosine Lateral Truncation Roll-off`):**
   داخل [`filter_weighted_sinogram_2d`](../src/highpitchct/recon_fbp.py)، وقبل تطبيق `FFT`، تُمدَّد حافتا الكاشف اليمنى واليسرى بعدد `lateral_pad_cols = 96` عمودًا تتناقص فيها القيمة بنعومة عبر $\frac{1}{2}\bigl(1 + \cos(\pi t)\bigr)$ من قيمة آخر عمود مقاس إلى `0.0`، تليها أصفار حتى قوة العدد `2` التالية (`next_pow2`) لمنع الالتفاف الدوري (`Circular Convolution Aliasing`) ومنع تقعّر الأطراف (`Truncation Cupping`).

---

## 3. الاشتقاق الرياضي الكامل لترجيح `Parker (1982)` الدقيق (`compute_parker_weights`)

ليكن مدى المسح القصير $\tilde{\beta} = \beta - \beta_0 \in [0, \pi + 2\delta]$ حيث $\delta = \frac{\beta_{N_\beta - 1} - \beta_0 - \pi}{2} \ge \gamma_{\max}$، ولتكن زاوية الشعاع الجانبية $\alpha_m = \arctan(u_m / d_0) \in [-\gamma_{\max}, +\gamma_{\max}]$.

تُعرّف الدالة [`compute_parker_weights`](../src/highpitchct/recon_fbp.py) الوزن $w(\tilde{\beta}, \alpha)$ في المناطق الثلاث لورقة `Parker (Medical Physics, 1982)`:

$$w(\tilde{\beta}, \alpha) = \begin{cases}
\sin^2\!\left(\dfrac{\pi}{4}\,\dfrac{\tilde{\beta}}{\delta - \alpha}\right), & 0 \le \tilde{\beta} \le 2(\delta - \alpha) \\[10pt]
1, & 2(\delta - \alpha) < \tilde{\beta} < \pi - 2\alpha \\[8pt]
\sin^2\!\left(\dfrac{\pi}{4}\,\dfrac{\pi + 2\delta - \tilde{\beta}}{\delta + \alpha}\right), & \pi - 2\alpha \le \tilde{\beta} \le \pi + 2\delta
\end{cases}$$

### برهان وحدة التجزئة للأشعة المرافقة (`Conjugate Partition of Unity`) وحلّ النقطتين الشاذتين `0/0`
ليكن الشعاع الأول في منطقة الدخول السفلية $\tilde{\beta}_1 \in [0, 2(\delta - \alpha)]$. شعاعه المرافق هو $\tilde{\beta}_2 = \tilde{\beta}_1 + \pi + 2\alpha$ بزاوية كاشف $\alpha_2 = -\alpha$. نلاحظ أن:

$$\tilde{\beta}_2 \in [\pi + 2\alpha,\; \pi + 2\delta] = [\pi - 2\alpha_2,\; \pi + 2\delta]$$

أي أن الشعاع المرافق يقع حتمًا في منطقة الخروج العلوية! وبالتعويض المباشر في الصيغة الثالثة مع $\delta + \alpha_2 = \delta - \alpha$ و$\pi + 2\delta - \tilde{\beta}_2 = 2(\delta - \alpha) - \tilde{\beta}_1$:

$$w(\tilde{\beta}_2, -\alpha) = \sin^2\!\left(\frac{\pi}{4}\,\frac{2(\delta - \alpha) - \tilde{\beta}_1}{\delta - \alpha}\right) = \sin^2\!\left(\frac{\pi}{2} - \frac{\pi}{4}\,\frac{\tilde{\beta}_1}{\delta - \alpha}\right) = \cos^2\!\left(\frac{\pi}{4}\,\frac{\tilde{\beta}_1}{\delta - \alpha}\right)$$

ومنه يتطابق المجموع جبريًا مع الواحد الصحيح في كل نقطة:

$$w(\tilde{\beta}, \alpha) + w(\tilde{\beta} + \pi + 2\alpha, -\alpha) = \sin^2(\eta) + \cos^2(\eta) \equiv 1$$

**معالجة الركنين الشاذين `(0, +\delta)` و`(\pi + 2\delta, -\delta)`:**
عندما يتساوى نصف المدى الزاوي مع أقصى زاوية مروحة ($\delta = \gamma_{\max}$ تمامًا كما في `short_scan` القياسي لدينا)، ينعدم المقام $\delta - \alpha$ عند العمود الأيمن الأقصى $\alpha = +\delta$ وعند $\tilde{\beta} = 0$ فتظهر صيغة غير معيّنة $0/0$ (ومثلها عند $\alpha = -\delta, \tilde{\beta} = \pi + 2\delta$). بتعيين القيمة الحدية الدقيقة **$w = 0.5$** في هاتين النقطتين الشاذتين:
1. يتحقق شرط المرافق: $w(0, +\delta) + w(\pi + 2\delta, +\delta \to -\delta) = 0.5 + 0.5 = 1.0$.
2. يتحقق تكامل العمود العددي (`Trapezoidal Quadrature`): مجموع أوزان أي عمود مضروبًا في $\Delta\beta$ يساوي **$\pi$ بدقة الآلة (`< 1e-13`)** عبر جميع أعمدة الكاشف من الحافة إلى الحافة!
3. أما في وضع المسح الكامل `full_scan` ($2\pi$)، فكل شعاع يُقاس مرتين بالضبط عبر الدورة الكاملة، فيكون الوزن الدقيق ثابتًا $w \equiv \frac{1}{2}$.

---

## 4. الاشتقاق الرياضي الدقيق لـ `Equispaced Collinear-Detector Fan-Beam FBP` ومعايرة `0.0 HU`

### 4.1 معادلة إعادة البناء لكاشف مسطح متساوي التباعد (`Kak & Slaney`, القسم 3.4.2)
في `Stage 2`، يقع الكاشف المسطح الافتراضي على المسافة الفيزيائية $d_0 = \text{SDD}$ من المنبع، بينما تكتب المراجع الكلاسيكية (`Kak & Slaney`) الكاشف منكمشًا إلى مركز الدوران على المسافة $\rho_0 = \text{SID}$ بإحداثي $u_{\text{iso}} = \frac{\rho_0}{d_0} u$ وخطوة $\Delta u_{\text{iso}} = \frac{\rho_0}{d_0} d_u$. عند تعويض هذا التحويل في صيغة جاكوبي المروحة المسطحة، **يختزل عامل التكبير $\rho_0 / d_0$ تمامًا بدون أي معامل تخميني**، فنحصل على الصيغة الفيزيائية المغلقة المُنفّذة في [`reconstruct_fanbeam_fbp`](../src/highpitchct/recon_fbp.py):

1. **الترجيح الجيبي التمامي المسبق وترجيح الوفرة الزاوية:**
   $$\tilde{g}(\beta_k, u_m) = g_{\text{completed}}(\beta_k, u_m) \cdot w_{\text{Parker}}(\beta_k, u_m) \cdot \frac{d_0}{\sqrt{d_0^2 + u_m^2}}$$

2. **الفلترة المكانية الدقيقة بمرشح `Ram-Lak` المنفصل (`Exact Spatial Impulse Response`):**
   بدلًا من كتابة $|f|$ ساذجةً في فضاء التردد (`|fftfreq|`) — والتي تجعل المركبة الصفرية $H[0] = 0$ على مصفوفة محدودة الطول فتحذف متوسط الإشارة وتُحدث **انحيازًا سالبًا وتقعّرًا (`~ -10 HU DC Bias`)** في الماء — نستخدم الاستجابة النبضية التحليلية الدقيقة في الفضاء الحقيقي (`Kak & Slaney`, المعادلة 61) بخطوة $\Delta u_{\text{iso}} = \frac{\rho_0}{d_0} d_u$:
   $$k[n] = \begin{cases}
   \dfrac{1}{4\,\Delta u_{\text{iso}}}, & n = 0 \\[10pt]
   0, & n \ne 0 \text{ زوجي} \\[8pt]
   -\dfrac{1}{\pi^2 n^2\,\Delta u_{\text{iso}}}, & n \text{ فردي}
   \end{cases} \qquad \bigl[\text{الوحدة الفيزيائية: }\text{mm}^{-1}\bigr]$$
   نضع $k[n]$ في مصفوفة دورية متناظرة طولها $N_{\text{fft}} = 2^{\lceil \log_2(2(N_u + 2 N_{\text{pad}}))\rceil}$، ونأخذ تحويل فورييه الحقيقي `H_ramp = rfft(k)` مضروبًا في نافذة التنعيم المختارة $W(f / f_c)$ (`hann`, `hamming`, `cosine`, `shepp_logan`, أو `ramp` الخام) مع الحفاظ الصارم على $W(0) = 1$:
   $$Q(\beta_k, u_m) = \Delta u_{\text{iso}} \sum_{m'} \tilde{g}(\beta_k, u_{m'})\, k_{\text{win}}[m - m'] = \operatorname{irfft}\!\Bigl(\operatorname{rfft}(\tilde{g}_{\text{padded}}) \cdot \Delta u_{\text{iso}}\, H_{\text{ramp}} \cdot W\Bigr)$$
   هذه الصيغة تُحقّق معايرة فيزيائية مذهلة: أسطوانة الماء المتجانسة تُعاد بناؤها إلى **`0.0 HU` ($\pm 0.3\;\text{HU}$)** في كلٍّ من `short_scan` و`full_scan` دون أي معامل تصحيح يدوي!

3. **الإسقاط الخلفي المرجّح بمقلوب مربع المسافة (`Weighted Fan-Beam Backprojection`):**
   في جملة الجانتري لدينا، تقع البؤرة الاسمية عند $\vec{F}_0(\beta) = (-\rho_0\sin\beta,\; +\rho_0\cos\beta)$، ويتجه الشعاع المركزي وفق $\hat{e}_{\text{ray}}(\beta) = (\sin\beta,\; -\cos\beta)$، ويتجه محور أعمدة الكاشف وفق $\hat{e}_u(\beta) = (\cos\beta,\; \sin\beta)$. لأي بكسل $(x, y)$، فإن مسقط متجه الموضع $\vec{r} - \vec{F}_0(\beta) = (x + \rho_0\sin\beta,\; y - \rho_0\cos\beta)$ يعطي:
   $$D(x, y; \beta) = (\vec{r} - \vec{F}_0)\cdot\hat{e}_{\text{ray}} = \rho_0 + x\sin\beta - y\cos\beta, \qquad U(x, y; \beta) = \frac{D(x, y; \beta)}{\rho_0}$$
   $$u^\star(x, y; \beta) = d_0\,\frac{(\vec{r} - \vec{F}_0)\cdot\hat{e}_u}{D(x, y; \beta)} = d_0\,\frac{x\cos\beta + y\sin\beta}{\rho_0 + x\sin\beta - y\cos\beta}, \qquad m^\star = \frac{u^\star}{d_u} + \frac{N_u - 1}{2}$$
   وباستيفاء $Q(\beta_k, m^\star)$ خطيًا عبر الأعمدة، يُجمع معامل التوهين الخطي $\mu(x, y)\;[\text{mm}^{-1}]$:
   $$\mu(x, y) = \sum_{k=0}^{N_\beta - 1} \frac{1}{U(x, y; \beta_k)^2}\, Q\bigl(\beta_k, u^\star(x, y; \beta_k)\bigr)\,\Delta\beta$$
   حيث خطوة التكامل الزاوي هي $\Delta\beta = \frac{\beta_{N_\beta - 1} - \beta_0}{N_\beta - 1}$ في `short_scan` (شبكة شاملة للطرفين) و$\Delta\beta = \frac{2\pi}{N_\beta}$ في `full_scan` الدوري، ثم يُحوّل $\mu$ إلى وحدات هاونسفيلد بالمعادلة القياسية:
   $$\text{HU}(x, y) = 1000 \cdot \frac{\mu(x, y) - \mu_{\text{water}}}{\mu_{\text{water}}}$$

---

## 5. المقاييس الكمية في التصوير المقطعي (`RMSE`, `MAE`, `PSNR`, و`SSIM` الفيزيائي غير السالب)

تُوفّر الدالة [`compute_reconstruction_metrics`](../src/highpitchct/recon_fbp.py) المقاييس الأربعة المعتمدة في تقييم إعادة البناء المقطعي داخل `fov_mask`:
- **`RMSE [HU]`** و**`MAE [HU]`** و**`MaxAbs [HU]`** مباشرةً بوحدات هاونسفيلد.
- **`PSNR [dB]`** على مدى ديناميكي قياسي `data_range_hu = 2000.0 HU` (من الهواء `-1000 HU` إلى العظم القشري `+1000 HU`).
- **`SSIM` (`compute_ssim_2d` بنافذة غاوسية قياسية $11 \times 11$ و$\sigma = 1.5$ وفق `Wang et al., 2004`):**
  ملاحظة فيزيائية جوهرية: عامل الإضاءة في `SSIM` $\frac{2\mu_x\mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1}$ مشتق حصريًا لإشارات **غير سالبة ($\ge 0$)**! إذا أُدخلت وحدات `HU` الموقّعة مباشرةً، يقع الماء والأنسجة الرخوة (`0 HU`) عند الصفر الحسابي بحيث يؤدي أي تموج طفيف ($\pm 15\;\text{HU}$) إلى $\mu_x \mu_y < 0$ وقيم `SSIM` سالبة مضللة. لذلك نُطبّق إزاحة هاونسفيلد الفيزيائية القياسية (المعتمدة في تحدي `AAPM Mayo LDCT`):
  $$\text{HU}_{\text{mod}} = \max(\text{HU} + 1000,\; 0) = 1000\,\frac{\mu}{\mu_{\text{water}}} \ge 0$$
  حيث يقابل الهواء (`-1000 HU`, $\mu = 0$) القيمة `0.0`، والماء (`0 HU`) القيمة `1000.0`، والعظم (`+1000 HU`) القيمة `2000.0`.

---

## 6. ملاحظة حتمية العتاد عبر المعالجات (`Xeon AVX-512 1-ULP Boundary Fix` في `Stage 2`)

عند تشغيل حزمة اختبارات `v0.5.0` على معالج `Intel Xeon` في `Kaggle`، كشف اختبار المطابقة الصارم `test_vectorized_vs_scalar_curved_to_flat_and_ssrb_parity` فرقًا دقيقًا في `21` عنصرًا حدوديًا من أصل `1512` مع بقاء `measured_mask` متطابقًا 100%:
- **السبب الجذري الدقيق:** في الحالة `Group 0` من البؤرة المتحركة حيث $\Delta\rho = 0$ ($\rho_g = \rho_0$)، يكون العمود المطابق نظريًا هو $c_g \equiv m$ بالضبط (`0.0` عند $m=0$ و`35.0` عند $m=35$). لكن حساب السلسلة المتسامية `d0 * tan(asin(sin(atan2(u, d0))))` عبر تعليمات المتجهات `Intel SVML AVX-512` أنتج تقريبًا عائمًا بمقدار `1 ULP` (`-3.55e-15 < 0.0`) أسقط العمود الحدّي من شرط `c_g >= 0.0` في الكود المتجهي دون الكود القياسي `math.*`، فانتقل ذلك البكسل الحدّي إلى `Group 1` ($\Delta\rho = +5.45\;\text{mm}$).
- **الإصلاح الجذري الدائم في [`rebinning.py`](../src/highpitchct/rebinning.py):** فرض المتطابقة الجبرية الصريحة $c_g \equiv m$ عندما $|\rho_g - \rho_0| \le 10^{-12}$ دون مرور بدوال مثلثية متعاكسة، مع هامش تسامح حدودي موحّد `1e-11` وهسترة كسر تعادل `abs_v < best_abs_v - 1e-11`، مما يضمن تطابقًا حتميًا عبر جميع معماريات `x86_64 / AVX2 / AVX-512 / ARM / GPU`.

---

## 7. البنية البرمجية لـ `Stage 3` وبوابة المقارنة البصرية الإلزامية

1. **[`src/highpitchct/recon_fbp.py`](../src/highpitchct/recon_fbp.py):**
   - `extrapolate_sinogram`: استكمال الفجوات بالأشعة المرافقة المقاسة والدمج السلس مع صون `measured_mask`.
   - `compute_parker_weights`: أوزان `Parker` الدقيقة للمسح القصير مع معالجة الركنين الشاذين (`0.5`).
   - `build_fbp_filter_kernel_rfft` و`filter_weighted_sinogram_2d`: المرشح المكاني الدقيق `Ram-Lak` مع نوافذ التنعيم والتمديد الجانبي بجيب التمام المرفوع.
   - `reconstruct_fanbeam_fbp` و`reconstruct_fanbeam_fbp_scalar_ref`: دعم صريح للـ Backends (`numpy`، `torch`، `torch_cuda` على `T4 GPU`) مع **حظر تام لأي انتقال صامت (`No silent fallback`)**.
   - `reconstruct_series_reference_streaming`: خط تدفقي كامل محدود الذاكرة من ملفات `DICOM-CT-PD` إلى شرائح $x_{\text{reference}}$ بذاكرة عشوائية منخفضة تناسب `Kaggle`.
2. **[`src/highpitchct/visual_qa.py`](../src/highpitchct/visual_qa.py) (لوحات ذات 6 نوافذ عالية الدقة):**
   - **`stage3_phantom_fbp_visual_qa.png`:** تقارن جنبًا إلى جنب بين (A) الأصل التحليلي الدقيق `Ground Truth`، (B) مرجعنا $x_{\text{reference}}$ عند `pitch=0.85`، (C) خريطة الفرق `FBP - GT` لإثبات انعدام التقعّر ودقة المعايرة، (D) إعادة بناء `High-Pitch (pitch=1.85)` الخام المثقلة بخطوط الفجوات (`inpaint='none'`)، (E) إعادة بناء `High-Pitch` بعد الاستكمال الفيزيائي `conjugate_feather`، و(F) قناع الاستكمال التعاوني وأوزان `Parker`.
   - **`stage3_<case_id>_fbp_visual_qa.png`:** تقارن للمريض الحقيقي `C012` بين (A) شريحة الصدر المنشورة من `Siemens` بنافذة الرئة، (B) مرجعنا $x_{\text{reference}}$ بنافذة الرئة `[-1000, 200] HU`، (C) مرجعنا $x_{\text{reference}}$ بنافذة المنصف والأنسجة الرخوة `[-160, 240] HU`، (D) تشوه `High-Pitch RDC` الخام دون استكمال، (E) تحسّن `High-Pitch RDC` مع `conjugate_feather`، و(F) الفرق الكمي بين حالة الـ `High-Pitch` ومرجعنا $x_{\text{reference}}$ — وهي بالضبط بقايا التشوه التي ستتعلّم الشبكة العصبية إزالتها في `Stage 7`!
3. **الأمر الموحّد في [`src/highpitchct/cli.py`](../src/highpitchct/cli.py):**
   ```bash
   hpct stage3-recon --case-id C012 --num-slices 5 --image-size 512 --filter hann --backend numpy --run-phantom-check --workers 8
   ```
   ويقبل أيضًا `--backend torch_cuda` عند تفعيل مسرّع `GPU T4 x2` على `Kaggle` لتنفيذ الإسقاط الخلفي والفلترة على البطاقة الرسومية بسرعة فائقة.
