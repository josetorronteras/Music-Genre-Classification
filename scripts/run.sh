#!/bin/bash
# Demo-menu shell script
## ----------------------------------
# Define variables
# ----------------------------------
EDITOR=nano
PASSWD=/etc/passwd
RED='\033[0;41;30m'
STD='\033[0;0;39m'
 
eval "$(conda shell.bash hook)"
conda activate tfg

# ----------------------------------
# User defined function
# ----------------------------------
pause(){
  read -p "Presiona [Enter] para continuar..." fackEnterKey
}

one(){
	python source/main.py --preprocess=spec --config=config/config-gpu.ini
	python source/main.py --preprocess=mfcc --config=config/config-gpu.ini
}
 
# do something in two()
two(){
	echo "two() called"
        pause
}
 
# menu
show_menus() {
	clear
	echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"	
	echo "  Clasificación de los géneros musicales  "
	echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
	echo "1. Extracción de características"
	echo "2. Creacción de los datasets"
	echo "3. Exit"
}
# Lee la accion sobre el teclado y la ejecuta.
# Invoca el () cuando el usuario selecciona 1 en el menú.
# Invoca a los dos () cuando el usuario selecciona 2 en el menú.
# Salir del menu cuando el usuario selecciona 3 en el menú.
read_options(){
	local choice
	read -p "Enter choice [ 1 - 3] " choice
	case $choice in
		1) one ;;
		2) two ;;
		3) exit 0;;
		*) echo -e "${RED}Error...${STD}" && sleep 2
	esac
}
 
# ----------------------------------------------
# Trap CTRL+C, CTRL+Z and quit singles
# ----------------------------------------------
trap '' SIGINT SIGQUIT SIGTSTP
 
# -----------------------------------
# Main logic - infinite loop
# ------------------------------------
while true
do
 
	show_menus
	read_options
done