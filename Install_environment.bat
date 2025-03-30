conda env create -f PokerRL.yml
call conda activate PokerRL
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
call conda deactivate
pause