# Ideas & Future Work

## Perceptual Loss

MSE loss'un problemi: piksel piksel karşılaştırıyor → bulanık sonuç.

Perceptual loss: Resimleri VGG gibi pretrained CNN'den geçir, intermediate feature map'leri karşılaştır. "Piksel benziyor mu?" yerine "Feature'lar benziyor mu?" sorusu.

```
Predicted Image → VGG → Features_pred
Target Image    → VGG → Features_target

Loss = MSE(Features_pred, Features_target)
```

VGG'nin conv katmanları texture, edge, shape gibi şeyleri encode ediyor. Bunları karşılaştırınca daha "insan gözüne uygun" benzerlik ölçülüyor.

Referans: "Perceptual Losses for Real-Time Style Transfer" - Johnson et al., 2016

---

## Diğer Fikirler

- [ ] Daha uzun eğitim (100+ epoch)
- [ ] MAE (Masked Autoencoder) tarzı: sadece visible patch'leri encode et
- [ ] Contrastive learning (SimCLR, DINO)
- [ ] Attention map visualization
- [ ] Gradio/Streamlit demo
