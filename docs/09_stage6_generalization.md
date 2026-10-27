# المرحلة السادسة (`Stage 6`): التدريب والتعميم متعدد المرضى/المناطق التشريحية مع ضمان عدم تسرب البيانات (`Zero-Leakage Out-of-Sample Generalization`)

## 1. الهدف العلمي والفيزيائي للمرحلة السادسة

بعد أن أثبتنا في **`Stage 5 (v0.8.2)`** قدرة الشبكة ثنائية المجال ذات المستويات الأربعة (`64 → 128 → 256 → 512`) والتدريب المتسلسل على 3 مراحل (`Eq. 1 → Eq. 2 → Eq. 3`) على خفض خطأ إعادة البناء `RMSE [HU]` ورفع `SSIM` بشكل رتيب على المريض `C012` عبر $p \in \{1.5, 2.0, 2.5\}$، تأتي **`Stage 6 (v0.9.0)`** للإجابة عن السؤال الأهم في لجنة مناقشة الماستر:

> **هل الشبكة تتعلّم قانوناً فيزيائياً وتشريحياً عاماً لإكمال الفجوة الحلزونية المفقودة وإزالة خطوط الـ Streak Artifacts، أم أنها تحفظ الشريحة التي تدربت عليها فقط؟**

وللإجابة عن هذا السؤال بصرامة علمية، تطبّق الوحدة [`generalization.py`](../src/highpitchct/generalization.py) **بوابة التدقيق الصارمة لمنع تسرب البيانات (`Zero-Leakage Audit Gate`)** قبل بدء أي تدريب، وتُقيّم النموذج بأوزان مجمّدة (`Frozen Zero-Shot Inference`) على بيانات اختبار لم ترها الشبكة إطلاقاً في أي مرحلة من مراحل التدريب الثلاث.

---

## 2. بروتوكولا الفصل الخالي من التسرب (`Zero-Leakage Split Modes`)

### أ. الفصل على مستوى المرضى (`patient_split`)
عند توفر مريضين أو أكثر (أو في الفانتومات المتعددة ذات التشريح والـ `native_pitch` المختلف):
$$\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test}} = \emptyset, \qquad \mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset$$
- يتم استخراج شرائح التدريب حصرياً من مرضى التدريب $\mathcal{P}_{\text{train}}$ (مثل `C012`)، وتُختبر الشبكة بدون أي تعديل (`Zero-Shot`) على مريض اختبار مستقل $\mathcal{P}_{\text{test}}$ (مثل `C027`).

### ب. الفصل المحوري ذو الحزام العازل المخروطي الحلزوني (`axial_guardband`)
لتشغيل التعميم فوراً على مريض واحد محمّل على القرص (مثل `C012` الذي يمتد مسحه الحلزوني لمسافة $\sim 255\text{ mm}$ من قمة الرئة إلى الحجاب الحاجز):
1. **منطقة التدريب العلوية/الوسطى (`Cranial / Mid-Thorax Training Zone`):** تُستخرج منها شرائح التدريب $z_{\text{train}}$.
2. **الحزام العازل الميت للشعاع المخروطي الحلزوني (`Helical Cone-Beam Guard Band`):**
   $$\min_{z_{\text{tr}}, z_{\text{te}}} |z_{\text{tr}} - z_{\text{te}}| \ge \Delta z_{\text{guard}} \ge 45.0\text{ mm}$$
3. **شرط عدم تداخل الشرائح ثلاثية الأبعاد لـ `ITF` (`3D ITF Slab Disjointness`):**
   يتم تحديد نصف عرض شريحة الاسقاط ثلاثي الأبعاد $h_{\text{slab}} \le 0.45\,\Delta z_{\text{guard}}$ بحيث يتحقق دائماً:
   $$\text{slab\_gap\_mm} = \min |z_{\text{tr}} - z_{\text{te}}| - 2\,h_{\text{slab}} > 0\text{ mm}$$
   وبذلك لا تشترك شرائح التدريب وشرائح الاختبار في أي مسقط مخروطي ثلاثي الأبعاد.

---

## 3. التشغيل عبر سطر الأوامر (`CLI`)

```bash
python -m highpitchct.cli stage6-generalize \
  --data-root /kaggle/working/highpitchct-data \
  --case-id C012 \
  --split-mode axial_guardband \
  --train-slices-per-case 2 \
  --test-slices-per-case 1 \
  --min-guardband-mm 45.0 \
  --pitches 1.5,2.0,2.5 \
  --itf-ratio 0.60 \
  --sino-channels 64 \
  --img-channels 64 \
  --epochs 36 \
  --lr 1.5e-3 \
  --backend torch_cuda \
  --save-model \
  --no-progress
```
