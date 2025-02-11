from classes.Thingy52Client import Thingy52Client
from utils.utility import scan, find
import asyncio


async def main():
    my_thingy_addresses = ["CC:40:09:57:8F:26"] #indirizzo del dispositivo
    discovered_devices = await scan()
    my_devices = find(discovered_devices, my_thingy_addresses) #ricerca del mio dispositivo

    thingy52 = Thingy52Client(my_devices[0])
    await thingy52.connect() #connessione con il mio dispositivo
    thingy52.save_to(str(input("Enter recording name:"))) #salvataggio dei dati raccolti su un file
    await thingy52.receive_inertial_data()


if __name__ == '__main__':
    asyncio.run(main())