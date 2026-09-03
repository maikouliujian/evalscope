import json
import os


def convert_aisbench_jsonl_to_evalscope_jsonl(input_file, output_file, model_name="/data1/Deepseek-V3.2-INT8/"):
    """
    将 AISBench 的 JSONL 逐行转换为 EvalScope 的 JSONL 格式
    """
    if not os.path.exists(input_file):
        print(f"Error: Cannot find input file {input_file}")
        return

    count = 0
    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)

                # 1. 提取 AISBench 原始数据
                idx = item.get("id", 0)
                prediction = item.get("prediction", "")

                # 2. 构造 EvalScope 的消息单元
                message_obj = {
                    "id": str(idx),
                    "content": prediction,
                    "source": "generate",
                    "metadata": None,
                    "internal": None,
                    "role": "assistant",
                    "tool_calls": None,
                    "model": model_name
                }

                # 3. 构造 EvalScope 的单行结构
                evalscope_line_obj = {
                    "index": idx,
                    "model": model_name,
                    "model_output": {
                        "model": model_name,
                        "choices": [
                            {
                                "message": message_obj,
                                "stop_reason": "stop",
                                "logprobs": None
                            }
                        ],
                        "usage": {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                            "reasoning_tokens": None
                        },
                        "time": None,
                        "metadata": None,
                        "error": None
                    },
                    "messages": [message_obj],
                    "metadata": {}
                }

                # 4. 逐行写入输出文件，不使用缩进，保持一行一个 JSON
                outfile.write(json.dumps(evalscope_line_obj, ensure_ascii=False) + "\n")
                count += 1

            except json.JSONDecodeError as e:
                print(f"Skipping invalid line: {e}")
                continue

    print(f"--- 转换完成 ---")
    print(f"成功转换条数: {count}")
    print(f"结果已保存至: {output_file}")


# --- 运行 ---
if __name__ == "__main__":
    # 修改为你的实际文件名
    input_path = "./lcb_code_generation_v6.jsonl"
    output_path = "live_code_bench_v6.jsonl"

    convert_aisbench_jsonl_to_evalscope_jsonl(input_path, output_path)