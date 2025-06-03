import xml.etree.ElementTree as ET
from xml.dom import minidom
import glob
import os

def merge_and_sort_xml(input_folder, output_file, file_pattern="town05_short_r*.xml"):
    # 收集所有<route>元素
    all_routes = []

    # 获取所有匹配的XML文件
    input_files = glob.glob(os.path.join(input_folder, file_pattern))
    print(f"找到 {len(input_files)} 个文件: {[os.path.basename(f) for f in input_files]}")

    # 遍历每个文件并收集route
    for file in input_files:
        try:
            tree = ET.parse(file)
            root = tree.getroot()
            for route in root.findall("route"):
                all_routes.append(route)
            print(f"已读取文件: {os.path.basename(file)}")
        except ET.ParseError as e:
            print(f"错误：文件 {os.path.basename(file)} 解析失败 - {str(e)}")

    # 按id排序（假设id为整数）
    try:
        all_routes.sort(key=lambda x: int(x.get("id")))
    except (ValueError, TypeError) as e:
        print(f"警告：部分route的id非整数，改用字符串排序")
        all_routes.sort(key=lambda x: x.get("id", ""))

    # 创建新的根节点 <routes>
    merged_root = ET.Element("routes")

    # 按排序后的顺序添加route
    for route in all_routes:
        merged_root.append(route)

    # 转换为字符串并美化输出
    rough_xml = ET.tostring(merged_root, encoding="utf-8")
    parsed_xml = minidom.parseString(rough_xml)
    pretty_xml = parsed_xml.toprettyxml(indent="  ", encoding="utf-8")

    # 移除多余的空白行
    pretty_xml = b"\n".join([line for line in pretty_xml.split(b"\n") if line.strip()])

    # 写入文件
    with open(output_file, "wb") as f:
        f.write(pretty_xml)
    print(f"\n合并完成！输出文件: {output_file} （已按id排序）")

if __name__ == "__main__":
    input_folder = "."
    output_file = "merged_sorted_routes.xml"
    merge_and_sort_xml(input_folder, output_file)