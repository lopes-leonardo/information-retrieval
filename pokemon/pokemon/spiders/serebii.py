import scrapy

class SerebiiSpider(scrapy.Spider):
    name = "serebii"

    recovered = []
    pokemons = []

    def start_requests(self):
        urls = [
            'https://www.serebii.net/pokedex-sm/',
            'https://www.serebii.net/pokedex-swsh/',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        limit = 7 if response.url.endswith("-sm/") else 8
        base_url = "https://www.pokemon.com/us/pokedex/"
        # Find pokemon selects
        selects = response.css('#content main table tr td form select')
        selects = selects[0:limit]

        # For each select find all pokemons
        for select in selects:
            options = select.css("option")
            # For each option recover all pokemons
            for option in options:
                if "value" not in option.attrib.keys():
                    continue
                
                pokemon_link = option.attrib["value"]
                if pokemon_link == "#":
                    continue
                    
                inner_texts = option.css("::text").get().split(" ")
                try:
                    pokemon_id = inner_texts[0]
                    pokemon_name = " ".join(inner_texts[1:])

                    if pokemon_id not in self.recovered:
                        pokemon_data = {
                            "pid": int(pokemon_id),
                            "name": pokemon_name.strip().replace("\n", "").replace("\r", "")
                        }
                        self.pokemons.append(pokemon_data)
                        self.recovered.append(pokemon_data['pid'])
                        
                        # Tratamento do nome
                        name = "-".join(pokemon_data["name"].lower().replace(".", "").replace("'", "").replace(":", "").split(" "))
                        if name.endswith("-f"):
                            name = name.replace("-f", "-female")
                        elif name.endswith("-m"):
                            name = name.replace("-m", "-male")
                        yield scrapy.Request(f"{base_url}{name}", self.parse_pokemon, cb_kwargs=pokemon_data)
                except Exception as e:
                    print("AAAAAQUIIIIIIII", option, inner_texts)
                    raise e
        # for pokemon in pokemons:            
        #     yield scrapy.Request(f"{base_url}{pokemon['name']}", self.parse_pokemon, cb_kwargs=pokemon)

    def parse_pokemon(self, response, pid=None, name=None):
        description = " ".join([text.replace("\n", "").strip() for text in response.css("div.version-descriptions p::text").getall()])
        types_div = response.css("div.dtm-type")[0]
        types = types_div.css("ul li a::text").getall()

        yield {
            "id": pid,
            "name": name,
            "description": description,
            "type-main": types[0],
            "type-second": types[1] if len(types) == 2 else None
        }


