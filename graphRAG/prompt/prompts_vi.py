EXTRACT_SYSTEM_PROMPT = """Bạn là một chuyên gia phân tích dữ liệu có nhiệm vụ trích xuất thông tin từ văn bản để xây dựng một cơ sở dữ liệu đồ thị (knowledge graph)."""

EXTRACT_PROMPT = (
    "### Role:\n"
    "Bạn là một chuyên gia phân tích dữ liệu có nhiệm vụ trích xuất thông tin từ văn bản để xây dựng một cơ sở dữ liệu đồ thị (knowledge graph).\n"
    "\n"
    "### Instruction: \n"
    "- Dựa vào văn bản sau đây, hãy trích xuất tất cả các thực thể (entities) và các mối quan hệ (relationships) giữa chúng.\n"
    "- Đối với mỗi thực thể, bao gồm 'id' (tên thực thể), 'type' (ví dụ: Người, Địa điểm, Sự kiện, Tổ chức) và 'role' (vai trò hoặc chức năng, ví dụ: 'Nữ hoàng Anh').\n"
    "- Nếu không có vai trò nào áp dụng, để chúng là chuỗi rỗng ('').\n"
    "- Đầu ra trả về ở định dạng JSON như sau, hãy nhớ rằng đầu ra trả về nằm trong thẻ <output>:\n"
    "<output>\n"
    '{{"nodes": [{{"id": "entity_name", "type": "entity_type", "role": "entity_role"}}],\n'
    '"relationships": [{{"source": "entity_1", "target": "entity_2", "type": "relationship_type"}}]}}\n'
    "</output>"
    "\n\n"
    "### Thực hiện với đầu vào sau:\n"
    "<input>\n"
    "{text}\n"
    "</input>\n"
)

ANSWERING_SYSTEM_PROMPT = """Bạn là trợ lý chuyên gia trong việc mô tả thông tin đầu vào của người dùng dựa trên thông tin được cung cấp."""

ANSWERING_PROMPT = (
    "### Role:\n"
    "Bạn là một chuyên gia trong việc mô tả thông tin đầu vào của người dùng dựa trên thông tin được cung cấp.\n"
    "\n"
    "### Instruction: \n"
    "- Hãy chắc chắn rằng bạn sử dụng **List graph schemas** được cung cấp để mô tả truy vấn một cách chính xác.\n"
    "- **List graph schemas** là thông tin về các mối quan hệ và thuộc tính mà bạn có thể sử dụng để mô tả.\n"
    "- Đầu ra trả về ở định dạng JSON như sau, hãy nhớ rằng đầu ra trả về nằm trong thẻ <output>:\n"
    "<output>\n"
    '{{"response": "your_response", "additional_data": "your_additional_data"}}\n'
    "</output>"
    "\n\n"
    "### List graph schemas:\n"
    "{schemas}"
    "\n\n"
    "### Thực hiện phản hồi với 'List graph schemas' được cung cấp:\n"
    "<input>\n"
    "{query}\n"
    "</input>\n"
)

GET_RELATIONSHIPS_CYPHER = (
    "MATCH (n)-[r]->(m)\n"
    "RETURN n.id AS source, type(r) AS relationship, m.id AS target\n"
)

GET_NODE_RELATIONSHIPS_CYPHER = (
    "MATCH (n)-[r]->(m)\n"
    "RETURN n.id AS source_id, n.name AS source_name, type(r) AS rel_type,\n"
    "       m.id AS target_id, m.name AS target_name, labels(n) AS source_labels, labels(m) AS target_labels\n"
)