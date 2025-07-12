from openai import OpenAI
from typing import Optional

from graphRAG.utils.utils import parse_json
from graphRAG.config.setting import llm_config
from graphRAG.prompt.prompts import EXTRACT_SYSTEM_PROMPT, EXTRACT_PROMPT

class GraphExtractor:
    def __init__(self, llm: Optional[OpenAI] = None):
        if llm is None:
            llm = OpenAI(api_key=llm_config.api_key, base_url=llm_config.base_url)

        self.llm = llm
        self.prompt_template = EXTRACT_PROMPT

    def _inject_prompt(self, text: str) -> str:
        prompt_str = self.prompt_template.format(
            text=text,
        )
        return prompt_str

    def extract_graph(self, doc: str) -> dict:
        nodes = []
        relationships = []
        
        prompt_str = self._inject_prompt(doc)
        
        response = self.llm.chat.completions.create(
            seed=llm_config.seed,
            temperature=llm_config.temperature,
            top_p=llm_config.top_p,
            model=llm_config.model,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_str},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        output_structure = parse_json(content)
        print(output_structure)

        nodes.extend(output_structure.get("nodes", []))
        relationships.extend(output_structure.get("relationships", []))


        return {"nodes": nodes, "relationships": relationships}
    
# if __name__ == "__main__":
#     extractor = GraphExtractor()
#     context = "Elizabeth I (7 September 1533 – 24 March 1603) was Queen of England and Ireland from 17 November 1558 until her death in 1603. She was the last and longest reigning monarch of the House of Tudor. Her eventful reign, and its effect on history and culture, gave name to the Elizabethan era.\nElizabeth was the only surviving child of Henry VIII and his second wife, Anne Boleyn. When Elizabeth was two years old, her parents' marriage was annulled, her mother was executed, and Elizabeth was declared illegitimate. Henry restored her to the line of succession when she was 10, via the Third Succession Act 1543. After Henry's death in 1547, Elizabeth's younger half-brother Edward VI ruled until his own death in 1553, bequeathing the crown to a Protestant cousin, Lady Jane Grey, and ignoring the claims of his two half-sisters, the Catholic Mary and the younger Elizabeth, in spite of statutes to the contrary. Edward's will was set aside within weeks of his death and Mary became queen, deposing and executing Jane. During Mary's reign, Elizabeth was imprisoned for nearly a year on suspicion of supporting Protestant rebels.\nUpon her half-sister's death in 1558, Elizabeth succeeded to the throne and set out to rule by good counsel. She depended heavily on a group of trusted advisers led by William Cecil, whom she created Baron Burghley. One of her first actions as queen was the establishment of an English Protestant church, of which she became the supreme governor. This arrangement, later named the Elizabethan Religious Settlement, would evolve into the Church of England. It was expected that Elizabeth would marry and produce an heir; however, despite numerous courtships, she never did. Because of this she is sometimes referred to as the \"Virgin Queen\". She was eventually succeeded by her first cousin twice removed, James VI of Scotland.\nIn government, Elizabeth was more moderate than her father and siblings had been. One of her mottoes was video et taceo (\"I see and keep silent\"). In religion, she was relatively tolerant and avoided systematic persecution. After the pope declared her illegitimate in 1570, which in theory released English Catholics from allegiance to her, several conspiracies threatened her life, all of which were defeated with the help of her ministers' secret service, run by Sir Francis Walsingham. Elizabeth was cautious in foreign affairs, manoeuvring between the major powers of France and Spain. She half-heartedly supported a number of ineffective, poorly resourced military campaigns in the Netherlands, France, and Ireland. By the mid-1580s, England could no longer avoid war with Spain.\nAs she grew older, Elizabeth became celebrated for her virginity. A cult of personality grew around her which was celebrated in the portraits, pageants, and literature of the day. The Elizabethan era is famous for the flourishing of English drama, led by playwrights such as William Shakespeare and Christopher Marlowe, the prowess of English maritime adventurers, such as Francis Drake and Walter Raleigh, and for the defeat of the Spanish Armada. Some historians depict Elizabeth as a short-tempered, sometimes indecisive ruler, who enjoyed more than her fair share of luck. Towards the end of her reign, a series of economic and military problems weakened her popularity. Elizabeth is acknowledged as a charismatic performer (\"Gloriana\") and a dogged survivor (\"Good Queen Bess\") in an era when government was ramshackle and limited, and when monarchs in neighbouring countries faced internal problems that jeopardised their thrones. After the short, disastrous reigns of her half-siblings, her 44 years on the throne provided welcome stability for the kingdom and helped to forge a sense of national identity.\n\n\n== Early life ==\n\nElizabeth was born at Greenwich Palace on 7 September 1533 and was named after her grandmothers, Elizabeth of York and Lady Elizabeth Howard. She was the second child of Henry VIII of England born in wedlock to survive infancy."
#     graph_data = extractor.extract_graph(context)
#     print(graph_data)