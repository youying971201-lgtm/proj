# Stage-5 視覺對照結果

本目錄保留從[團隊的 Stage-5 成果資料夾](https://drive.google.com/drive/folders/1Lkz7OclwfAnIYCX34BV7lVHHbbSDUK7R)下載的四張代表性對照圖，涵蓋白天、夜間與路口標線。圖中比較原始 28k checkpoint 與 Stage-5 best（圖面標示 iter 2000）的分割疊圖；詳細圖另附兩者遮罩與差異。這些是**定性視覺結果**，不能替代 mIoU 等量化評估。

| 圖片 | 用途 |
| --- | --- |
| [三種路況總覽](three_images_28k_vs_stage5/three_images_overlay_comparison.jpg) | 將來源影像、原始 28k 與 Stage-5 疊圖並排。 |
| [白天道路詳細對照](three_images_28k_vs_stage5/per_image/1774-1/checkpoint_comparison.jpg) | 檢視白天車道線的分割與遮罩差異。 |
| [夜間道路詳細對照](three_images_28k_vs_stage5/per_image/night_56400/checkpoint_comparison.jpg) | 檢視夜間標線的分割與遮罩差異。 |
| [路口標線詳細對照](three_images_28k_vs_stage5/per_image/sidewalk_2/checkpoint_comparison.jpg) | 檢視停止線、斑馬線與車道線的分割差異。 |

本機只保留以上四張重要驗證圖；原始成果資料夾中的其他訓練過程圖片仍可由上述 Google Drive 連結取得。
