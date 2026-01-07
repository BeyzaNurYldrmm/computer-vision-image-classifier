python -m venv .venv
.venv\Scripts\activate

python -m pip install --upgrade pip

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

Write-Host "Kurulum tamamlandı"
Write-Host "Calistirmak icin:"
Write-Host "cd proje_dizin_yolu"
Write-Host ".venv\Scripts\activate"
Write-Host "python train.py --config configs/model_name(vit,beit,efficientnet,swin,convnext).toml"
