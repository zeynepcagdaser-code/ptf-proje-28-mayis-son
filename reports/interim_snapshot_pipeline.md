# Point-in-Time Interim MCP Snapshot Pipeline

## Amaç

Bu adımın amacı EPİAŞ `interim-mcp` endpointinden gerçek point-in-time K.PTF snapshot arşivi oluşturmaktır. Bu arşiv ileride leak-free backtest için kullanılabilir; bu aşamada finalized MCP ile join, feature engineering veya model eğitimi yapılmaz.

## Kapsam

Eklenen script:

- `snapshot_interim_mcp.py`

Çıktı CSV:

- `data/snapshots/interim_mcp_snapshots.csv`

Raporlar:

- `reports/interim_snapshot_pipeline.md`
- `reports/interim_snapshot_pipeline.json`

## Snapshot Semantiği

Script historical backfill yapmaz. Her çalıştırmada yalnızca çalıştırma anında endpointte görünür olan bugün ve yarın teslim günlerini çeker.

Her satır şu point-in-time metadata ile saklanır:

- `snapshot_ts`: snapshotın alındığı an
- `fetch_run_id`: run bazlı benzersiz id
- `delivery_date`: EPİAŞ response içindeki teslim tarihi
- `delivery_hour`: EPİAŞ response içindeki teslim saati
- `marketTradePrice`: endpointin o snapshot anında döndürdüğü K.PTF değeri
- `published_status_completed`: varsa `interim-mcp-published-status` sonucu
- `response_hash`: canonical payload + response items SHA-256 hash değeri
- `source_endpoint`: kullanılan endpoint

## Leakage Kontrolleri

Bu pipeline özellikle önceki oracle leakage bulgusunu tekrar üretmemek için dar tutuldu.

- finalized MCP ile join yok
- historical backfill yok
- eski snapshot overwrite yok
- her run yeni `snapshot_ts` ve `fetch_run_id` üretir
- endpoint response hashlenir
- duplicate snapshot yalnızca aynı `fetch_run_id`, `delivery_date`, `delivery_hour`, `response_hash` anahtarıyla tekrarlandıysa elenir

Bu yüzden aynı teslim saati için farklı zamanlarda alınan snapshotlar korunur. Bu davranış bilinçli: correction forecasting için asıl değer snapshot versiyonları arasındaki değişim ve finalized değere göre sonradan ölçülecek farktır.

## Operasyonel Tasarım

Script mevcut TGT login düzenini kullanır. Network timeout ve HTTP retry mantığı vardır:

- timeout/network hatalarında exponential backoff
- `429` durumunda daha uzun bekleme
- `5xx` durumunda retry
- lock file ile eşzamanlı çift çalışmayı engelleme
- CSV yazımında atomic temp-file replacement

CSV append-only mantığı pratikte şöyle uygulanır:

1. Mevcut snapshot CSV okunur.
2. Yeni run satırları oluşturulur.
3. Sadece birebir duplicate run/hash satırları temizlenir.
4. Tüm snapshot arşivi atomic olarak yeniden yazılır.

Bu CSV seviyesinde basit ve denetlenebilir bir başlangıç sağlar. Büyük hacimde canonical storage olarak partitioned parquet daha doğru olacaktır.

## GitHub Actions Önerisi

Eklenen öneri workflow:

- `.github/workflows/snapshot_interim_mcp.yml`

Başlangıç için saatlik çalışma uygundur. Daha hedefli bir program istenirse DAM yayın penceresi etrafında günde birkaç yoğun snapshot alınabilir. Örneğin:

- yayın öncesi
- yayın anı civarı
- yayın sonrası birkaç saat
- gün sonu kontrol snapshotı

Saatlik çalışma daha fazla veri üretir ama publication/revision zamanlarını kaçırma riskini azaltır.

## Storage Stratejisi

Başlangıçta CSV tercih edilebilir çünkü:

- Git diff ile kolay denetlenir
- küçük hacimde basit operasyon sağlar
- workflow commit/push mekanizmasıyla uyumludur

Ancak uzun vadede CSV canonical store olmamalıdır. Saatlik snapshot birkaç ay içinde büyür; her append tüm dosyayı yeniden yazdığı için corruption blast radius ve commit boyutu artar.

Önerilen uzun vadeli yapı:

- canonical: partitioned parquet
- partition keys: `snapshot_date`, `delivery_date`
- CSV: sadece hafif export veya son dönem görünümü

Parquet avantajları:

- daha küçük dosya
- hızlı tarama
- partition bazlı sınırlı rewrite
- future feature pipeline için daha iyi reproducibility

## Sonraki Güvenli Adım

Correction modeline geçmeden önce snapshot pipeline birkaç hafta çalıştırılmalı. Ardından ayrı bir audit ile şu sorular cevaplanmalı:

- erken snapshot K.PTF ile finalized PTF arasında gerçekten fark var mı?
- fark hangi saatlerde/hangi yayın pencerelerinde oluşuyor?
- `published_status_completed` correction variance ile ilişkili mi?
- aynı delivery hour için snapshotlar arasında revizyon var mı?

Bu cevaplar gelmeden correction forecasting backtesti güvenilir kabul edilmemelidir.
