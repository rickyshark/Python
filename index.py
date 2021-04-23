import reconocimiento_senas as ReconocimientoSenas

print('Eliga una Opcion: \n 1 - Reconocimiento de Senas \n 2 - Reconocimiento Facial \n 3 - Salir')
Entrada = int(input())

if(Entrada == 1):
 ReconocimientoSenas.comenzarReconocimiento()

