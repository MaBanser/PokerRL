echo off
echo "Removing existing PokerRL Environment"
call conda remove --name PokerRL --all
echo "Creating Conda Environment:"
echo:
call conda env create -f PokerRL.yml
echo "Environment successfully created"
echo:
echo "Installing Pytorch"
echo:
call conda activate PokerRL
call pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
call conda deactivate
echo:
echo "Done"
pause