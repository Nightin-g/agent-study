// 这个文件负责浏览器端交互：
// 1. 读取输入框里的 prompt。
// 2. 调用后端 POST /api/chat。
// 3. 把模型回答和工具调用轨迹显示到页面上。

const chatForm = document.querySelector("#chatForm");
const messageInput = document.querySelector("#message");
const useToolsInput = document.querySelector("#useTools");
const temperatureInput = document.querySelector("#temperature");
const temperatureValue = document.querySelector("#temperatureValue");
const sendButton = document.querySelector("#sendButton");
const answerBox = document.querySelector("#answer");
const traceBox = document.querySelector("#trace");
const logPathBox = document.querySelector("#logPath");
const healthText = document.querySelector("#healthText");
const keyStatus = document.querySelector("#keyStatus");

// 滑动 temperature 时，同步更新右侧数字。
temperatureInput.addEventListener("input", () => {
  temperatureValue.textContent = temperatureInput.value;
});

// 监听表单提交。event.preventDefault() 会阻止浏览器刷新页面。
chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const message = messageInput.value.trim();
  if (!message) {
    answerBox.textContent = "请输入问题。";
    return;
  }

  setLoading(true);
  answerBox.textContent = "Thinking...";
  renderTrace([]);

  try {
    // fetch 用来调用后端接口。
    // 浏览器访问的是同一个本地服务，所以这里可以直接写 /api/chat。
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message,
        use_tools: useToolsInput.checked,
        temperature: Number(temperatureInput.value),
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "请求失败");
    }

    answerBox.textContent = data.answer || "模型没有返回内容。";
    renderTrace(data.tool_steps || []);
    renderLogPath(data.log_path);
  } catch (error) {
    answerBox.textContent = error.message;
    renderLogPath("");
  } finally {
    setLoading(false);
  }
});

async function loadHealth() {
  try {
    // 页面加载时先问后端当前配置状态。
    const response = await fetch("/api/health");
    const data = await response.json();

    healthText.textContent = `${data.model} @ ${data.base_url}`;
    keyStatus.textContent = data.has_api_key ? "Key 已配置" : "Key 未配置";
    keyStatus.classList.toggle("ok", Boolean(data.has_api_key));
  } catch {
    healthText.textContent = "Local service unavailable";
    keyStatus.textContent = "离线";
  }
}

function setLoading(isLoading) {
  sendButton.disabled = isLoading;
  sendButton.textContent = isLoading ? "请求中" : "发送";
}

function renderTrace(steps) {
  traceBox.innerHTML = "";

  if (!steps.length) {
    traceBox.className = "trace empty";
    traceBox.textContent = "暂无工具调用";
    return;
  }

  traceBox.className = "trace";

  // 每一个 step 对应一次工具调用。
  // 你可以从这里观察模型给工具传了什么参数，以及工具返回了什么结果。
  for (const step of steps) {
    const item = document.createElement("div");
    item.className = "trace-item";

    const title = document.createElement("div");
    title.className = "trace-name";
    title.textContent = step.name;

    const code = document.createElement("pre");
    code.className = "trace-code";
    code.textContent = JSON.stringify(
      {
        arguments: step.arguments,
        result: step.result,
      },
      null,
      2,
    );

    item.append(title, code);
  traceBox.append(item);
  }
}

function renderLogPath(logPath) {
  logPathBox.textContent = logPath ? `日志已写入 ${logPath}` : "";
}

loadHealth();
