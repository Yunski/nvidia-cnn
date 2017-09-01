import asyncio

async def nierp():
    print("nierp")

async def niep():
    await nierp()
    while True:
        print("niep")

if __name__ == '__main__':
    ioloop = asyncio.get_event_loop()
    ioloop.run_until_complete(niep())
    ioloop.close()
