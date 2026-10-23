# ADR-0003: الـ pitch خاص بكل Series

- **الحالة:** مقبول
- **القرار:** لا نثبت pitch على 0.9 أو 0.6. سنقدّره من fit قوي بين موضع الطاولة والزاوية unwrapped، ونقارنه بالـ header.
- **تفصيل لاحق:** table feed لكل دورة = `2π |a|` عندما `z0 = a φ + b`. normalized pitch = table feed / collimated coverage at isocenter. تحفظ إشارة `a` لاتجاه المسح.
- **تنبيه:** base table trajectory وFFS z-offset كميتان منفصلتان.
