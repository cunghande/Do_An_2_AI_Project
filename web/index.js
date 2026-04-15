const API_URL = "http://127.0.0.1:8000";
let currentSequence = []; // Khởi tạo mảng rỗng

// 1. Kiểm tra trạng thái API
async function checkStatus() {
    try {
        const r = await fetch(`${API_URL}/`);
        if(r.ok) {
            const statusText = document.getElementById('status-text');
            statusText.innerText = "API Online";
            statusText.className = "text-green-600 font-bold";
            fetchStats();
            // Load thử User 1 khi mới vào trang cho đẹp
            document.getElementById('userIdInput').value = 1;
            loadUserHistory();
        }
    } catch(e) { 
        console.error("Chưa bật Backend!"); 
    }
}

// 2. DASHBOARD: Hiển thị thông số Accuracy
async function fetchStats() {
    try {
        const r = await fetch(`${API_URL}/stats`);
        const d = await r.json();
        document.getElementById('hr-metric').innerText = d.results?.HR_10 || '0.198';
        document.getElementById('ndcg-metric').innerText = d.results?.NDCG_10 || '0.105';
    } catch(e) {}
}

// 3. LOGIC ĐĂNG NHẬP: Lấy lịch sử
async function loadUserHistory() {
    const userId = document.getElementById('userIdInput').value;
    if (!userId) return;

    console.log("🚀 Đang đăng nhập User:", userId);

    try {
        const response = await fetch(`${API_URL}/users/${userId}/history`);
        if (response.ok) {
            const data = await response.json();
            
            // CẬP NHẬT BIẾN TOÀN CỤC
            currentSequence = data.history.map(item => item.item_id);
            console.log("📜 Lịch sử mới nạp:", currentSequence);

            // Vẽ Timeline và GỌI AI NGAY LẬP TỨC
            renderTimeline(); 
        } else {
            alert("Không thấy User này!");
        }
    } catch (error) { 
        console.error("Lỗi kết nối!"); 
    }
}

// 4. GỌI MODEL AI: Tính toán theo chuỗi (Sequential Logic)
async function getRecommendations() {
    if (currentSequence.length === 0) return;

    console.log("🧠 Model GRU đang tính toán cho chuỗi:", currentSequence);
    
    // Hiện hiệu ứng chờ (Loader)
    document.getElementById('recommendation-grid').classList.add('opacity-20');
    document.getElementById('loader').classList.remove('hidden');
    
    try {
        const response = await fetch(`${API_URL}/recommend`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ 
                sequence_history: currentSequence, 
                top_k: 8 
            })
        });
        const data = await response.json();
        
        // CẬP NHẬT ẢNH MỚI LÊN MÀN HÌNH
        renderCards(data.recommendations);
    } catch (error) {
        console.error("Lỗi Model:", error);
    } finally {
        document.getElementById('loader').classList.add('hidden');
        document.getElementById('recommendation-grid').classList.remove('opacity-20');
    }
}

// 5. HIỂN THỊ SẢN PHẨM: Xóa món cũ, nạp món mới
function renderCards(items) {
    const grid = document.getElementById('recommendation-grid');
    grid.innerHTML = ""; // Xóa sạch đồ cũ

    grid.innerHTML = items.map(item => `
        <div class="bg-white border p-4 rounded-2xl shadow hover:shadow-lg transition animate-in zoom-in duration-300">
            <div class="relative h-32 bg-slate-100 rounded-xl mb-2 flex items-center justify-center overflow-hidden">
                <div class="absolute inset-0 flex items-center justify-center text-slate-300 font-black text-4xl opacity-20">
                    ID: ${item.item_id}
                </div>
                <img src="${item.img_url}?t=${Date.now()}" class="max-h-full object-contain z-10" 
                     onerror="this.src='https://via.placeholder.com/150?text=ID+${item.item_id}'">
            </div>
            <h4 class="text-[10px] font-bold truncate">ID: ${item.item_id} - ${item.title}</h4>
            <div class="flex justify-between items-center mt-2">
                <span class="text-blue-600 font-bold">$${item.price}</span>
                <span class="text-[9px] bg-slate-100 px-1 rounded">Score: ${item.score}</span>
            </div>
        </div>
    `).join('');
    lucide.createIcons();
}

// 6. VẼ TIMELINE & TỰ ĐỘNG GỌI GỢI Ý
function renderTimeline() {
    const container = document.getElementById('timeline');
    container.innerHTML = currentSequence.map((id, index) => `
        <div class="flex items-center gap-3 relative group">
            <div class="w-6 h-6 rounded-full bg-slate-800 text-white text-[9px] flex items-center justify-center shrink-0 z-10 font-bold">${index + 1}</div>
            ${index < currentSequence.length - 1 ? '<div class="absolute left-3 top-6 w-0.5 h-4 bg-slate-200"></div>' : ''}
            <div class="bg-slate-50 flex-1 px-3 py-1.5 rounded-lg border border-slate-100 text-[11px] font-bold flex justify-between items-center">
                ID: ${id}
                <button onclick="removeItem(${index})" class="text-slate-300 hover:text-red-500 transition"><i data-lucide="x" class="w-3 h-3"></i></button>
            </div>
        </div>
    `).join('');
    lucide.createIcons();
    
    // QUAN TRỌNG: Mỗi lần Timeline vẽ lại là phải gọi AI tính toán gợi ý mới ngay
    getRecommendations();
}

function addNewItem() {
    const val = document.getElementById('addItemInput').value;
    if(val) { 
        currentSequence.push(parseInt(val)); 
        renderTimeline(); 
        document.getElementById('addItemInput').value = ''; 
    }
}

function removeItem(index) {
    currentSequence.splice(index, 1);
    renderTimeline();
}

function closeModal() { 
    document.getElementById('modal').classList.add('hidden'); 
}

// Khởi chạy ban đầu
lucide.createIcons();
checkStatus();