# ADR-0002: مرجع التدريب من خط إعادة بناء موحد

- **الحالة:** مقبول
- **القرار:** نبني target الصورة من full-dose raw projections عند pitch الأصلي عبر خطنا. صور الجهاز المنشورة للمقارنة فقط.
- **السبب:** خوارزمية إعادة بناء الجهاز وkernel والمعالجة اللاحقة غير معروفة، وقد يتعلم النموذج تقليدها بدل تعويض high-pitch degradation.
- **المصطلح:** نستخدم Reference Reconstruction، لا Absolute Ground Truth.
