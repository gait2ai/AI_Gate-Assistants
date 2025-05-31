// static/js/app.js
/**
 * AI Gate Application - Frontend JavaScript
 * Enhanced version with improved error handling, retry logic, and user experience
 */

document.addEventListener('DOMContentLoaded', function () {
    // DOM Element References
    const chatMessagesContainer = document.getElementById('chatMessages');
    const messageInputElement = document.getElementById('messageInput');
    const sendButtonElement = document.getElementById('sendButton');
    const chatFormElement = document.getElementById('chatForm');
    const pageTitleElement = document.getElementById('pageTitle');
    const chatHeaderTextElement = document.getElementById('chatHeaderText');
    const institutionLogoElement = document.getElementById('institutionLogo');

    // API Endpoint URLs
    const BASE_API_URL = window.location.origin;
    const CHAT_API_URL = `${BASE_API_URL}/api/chat`;
    const INSTITUTION_API_URL = `${BASE_API_URL}/api/institution`;

    // Application State
    let chatHistory = [];
    let currentSessionId = null;
    let isFetchingResponse = false;
    let retryCount = 0;
    const MAX_RETRIES = 3;
    const RETRY_DELAY = 2000; // 2 seconds
    
    // Performance and UX improvements
    let lastMessageTime = 0;
    const MESSAGE_THROTTLE = 1000; // Prevent spam clicking
    let institutionData = null;

    // --- Utility Functions ---
    
    /**
     * Generates a UUID v4 for session identification
     * @returns {string} UUID v4 string
     */
    function generateUUIDv4() {
        return ([1e7] + -1e3 + -4e3 + -8e3 + -1e11).replace(/[018]/g, c =>
            (c ^ (crypto.getRandomValues(new Uint8Array(1))[0] & (15 >> (c / 4)))).toString(16)
        );
    }

    /**
     * Delays execution for specified milliseconds
     * @param {number} ms - Milliseconds to delay
     * @returns {Promise} Promise that resolves after delay
     */
    function delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Throttles function execution to prevent spam
     * @param {Function} func - Function to throttle
     * @param {number} limit - Time limit in milliseconds
     * @returns {Function} Throttled function
     */
    function throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        };
    }

    /**
     * Sanitizes HTML content for safe display
     * @param {string} html - HTML string to sanitize
     * @returns {string} Sanitized HTML string
     */
    function sanitizeHTML(html) {
        const div = document.createElement('div');
        div.innerHTML = html;
        
        // Remove potentially dangerous elements and attributes
        const dangerousElements = ['script', 'iframe', 'object', 'embed', 'form'];
        dangerousElements.forEach(tag => {
            const elements = div.querySelectorAll(tag);
            elements.forEach(el => el.remove());
        });
        
        // Remove dangerous attributes
        const allElements = div.querySelectorAll('*');
        allElements.forEach(el => {
            const dangerousAttrs = ['onclick', 'onload', 'onerror', 'onmouseover', 'onfocus'];
            dangerousAttrs.forEach(attr => {
                if (el.hasAttribute(attr)) {
                    el.removeAttribute(attr);
                }
            });
        });
        
        return div.innerHTML;
    }

    // --- Session Management ---
    
    /**
     * Initializes or retrieves session ID
     */
    function initializeSession() {
        try {
            const storedSessionId = sessionStorage.getItem('aiGateChatSessionId');
            if (storedSessionId && storedSessionId.length > 10) {
                currentSessionId = storedSessionId;
                console.info("AI Gate Session restored:", currentSessionId);
            } else {
                currentSessionId = generateUUIDv4();
                sessionStorage.setItem('aiGateChatSessionId', currentSessionId);
                console.info("AI Gate Session created:", currentSessionId);
            }
        } catch (error) {
            console.error("Session initialization error:", error);
            currentSessionId = generateUUIDv4(); // Fallback without storage
        }
    }

    // --- Institution Info Management ---
    
    /**
     * Fetches and applies institution information with retry logic
     */
    async function fetchAndSetInstitutionInfo() {
        const maxRetries = 3;
        let attempt = 0;
        
        while (attempt < maxRetries) {
            try {
                console.info(`Fetching institution info (attempt ${attempt + 1}/${maxRetries})`);
                
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
                
                const response = await fetch(INSTITUTION_API_URL, {
                    signal: controller.signal,
                    headers: {
                        'Accept': 'application/json',
                        'Cache-Control': 'no-cache'
                    }
                });
                
                clearTimeout(timeoutId);
                
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const data = await response.json();
                
                if (data && data.name) {
                    institutionData = data;
                    const institutionName = data.name;
                    
                    // Update page elements
                    if (pageTitleElement) {
                        pageTitleElement.textContent = `${institutionName} Assistant`;
                        document.title = `${institutionName} Assistant`;
                    }
                    if (chatHeaderTextElement) {
                        chatHeaderTextElement.textContent = `${institutionName} Assistant`;
                    }
                    if (institutionLogoElement) {
                        institutionLogoElement.alt = `شعار ${institutionName}`;
                    }
                    
                    console.info("Institution info loaded successfully:", institutionName);
                    return; // Success, exit retry loop
                    
                } else {
                    throw new Error("Invalid institution data structure");
                }
                
            } catch (error) {
                attempt++;
                console.warn(`Institution info fetch failed (attempt ${attempt}):`, error.message);
                
                if (attempt < maxRetries) {
                    await delay(1000 * attempt); // Progressive delay
                } else {
                    console.error("Failed to fetch institution info after all retries");
                    setFallbackInstitutionInfo();
                }
            }
        }
    }

    /**
     * Sets fallback institution information
     */
    function setFallbackInstitutionInfo() {
        const fallbackName = "AI Gate Assistant";
        if (pageTitleElement) {
            pageTitleElement.textContent = fallbackName;
            document.title = fallbackName;
        }
        if (chatHeaderTextElement) {
            chatHeaderTextElement.textContent = fallbackName;
        }
        if (institutionLogoElement) {
            institutionLogoElement.alt = "شعار المساعد الذكي";
        }
    }

    // --- Chat History Management ---
    
    /**
     * Loads chat history from session storage with error handling
     */
    function loadChatHistory() {
        try {
            const storedHistory = sessionStorage.getItem('aiGateChatHistory');
            if (!storedHistory) {
                console.info("No chat history found");
                return;
            }
            
            const parsedHistory = JSON.parse(storedHistory);
            if (!Array.isArray(parsedHistory)) {
                throw new Error("Invalid history format");
            }
            
            chatHistory = parsedHistory;
            chatMessagesContainer.innerHTML = '';
            
            let messagesRestored = 0;
            chatHistory.forEach((item, index) => {
                try {
                    if (item.type === 'user' && item.text) {
                        displayUserMessage(item.text, item.timestamp, false);
                        messagesRestored++;
                    } else if (item.type === 'assistant' && item.text) {
                        displayAssistantMessage(item.text, item.timestamp, false);
                        messagesRestored++;
                    }
                } catch (error) {
                    console.warn(`Error restoring message ${index}:`, error);
                }
            });
            
            if (messagesRestored > 0) {
                console.info(`Restored ${messagesRestored} messages from history`);
                scrollToBottom();
            }
            
        } catch (error) {
            console.error("Error loading chat history:", error);
            sessionStorage.removeItem('aiGateChatHistory');
            chatHistory = [];
        }
    }

    /**
     * Saves message to history with error handling
     */
    function saveMessageToHistory(type, text, timestamp) {
        try {
            // Skip saving typing indicators
            if (text.includes('loading-dots') && type === 'assistant') {
                return;
            }

            // Validate inputs
            if (!type || !text || typeof text !== 'string') {
                console.warn("Invalid message data for history save");
                return;
            }

            const messageData = {
                type: type,
                text: text,
                timestamp: timestamp || formatTimestampForDisplay(),
                id: generateUUIDv4() // Add unique ID for each message
            };

            chatHistory.push(messageData);

            // Limit history size to prevent storage overflow
            const MAX_HISTORY_SIZE = 100;
            if (chatHistory.length > MAX_HISTORY_SIZE) {
                chatHistory = chatHistory.slice(-MAX_HISTORY_SIZE);
            }

            sessionStorage.setItem('aiGateChatHistory', JSON.stringify(chatHistory));
            
        } catch (error) {
            console.error("Error saving to chat history:", error);
            
            // If storage is full, try to clear some space
            if (error.name === 'QuotaExceededError') {
                try {
                    chatHistory = chatHistory.slice(-50); // Keep only last 50 messages
                    sessionStorage.setItem('aiGateChatHistory', JSON.stringify(chatHistory));
                    console.info("Chat history trimmed due to storage limit");
                } catch (trimError) {
                    console.error("Failed to trim chat history:", trimError);
                    chatHistory = [];
                }
            }
        }
    }

    // --- UI Message Display ---
    
    /**
     * Formats timestamp for display
     */
    function formatTimestampForDisplay() {
        try {
            const now = new Date();
            return now.toLocaleTimeString('ar-EG', { 
                hour: 'numeric', 
                minute: '2-digit', 
                hour12: true 
            });
        } catch (error) {
            console.warn("Timestamp formatting error:", error);
            return new Date().toLocaleTimeString();
        }
    }

    /**
     * Displays user message with enhanced validation
     */
    function displayUserMessage(text, timestamp = null, save = true) {
        try {
            if (!text || typeof text !== 'string') {
                console.error("Invalid user message text");
                return;
            }

            const messageDiv = document.createElement('div');
            messageDiv.className = 'message user-message';
            messageDiv.setAttribute('role', 'log');
            messageDiv.setAttribute('aria-label', 'رسالة المستخدم');

            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = text.trim();
            messageDiv.appendChild(contentDiv);

            const timeDiv = document.createElement('div');
            timeDiv.className = 'message-time';
            const displayTimestamp = timestamp || formatTimestampForDisplay();
            timeDiv.textContent = displayTimestamp;
            timeDiv.setAttribute('aria-label', `وقت الإرسال: ${displayTimestamp}`);
            messageDiv.appendChild(timeDiv);

            chatMessagesContainer.appendChild(messageDiv);
            
            if (save) {
                saveMessageToHistory('user', text, displayTimestamp);
            }
            
            scrollToBottom();
            
        } catch (error) {
            console.error("Error displaying user message:", error);
        }
    }

    /**
     * Displays assistant message with improved sanitization
     */
    function displayAssistantMessage(htmlText, timestamp = null, save = true) {
        try {
            if (!htmlText || typeof htmlText !== 'string') {
                console.error("Invalid assistant message text");
                return;
            }

            const messageDiv = document.createElement('div');
            messageDiv.className = 'message assistant-message';
            messageDiv.setAttribute('role', 'log');
            messageDiv.setAttribute('aria-label', 'رد المساعد');

            const logoImg = document.createElement('img');
            logoImg.src = "assets/logo.png";
            logoImg.alt = institutionData ? `شعار ${institutionData.name}` : "شعار المساعد";
            logoImg.className = 'assistant-logo';
            logoImg.onerror = function() {
                this.style.display = 'none';
                console.warn("Assistant logo failed to load");
            };
            messageDiv.appendChild(logoImg);

            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            
            // Sanitize HTML content before inserting
            const sanitizedHTML = sanitizeHTML(htmlText);
            contentDiv.innerHTML = sanitizedHTML;
            messageDiv.appendChild(contentDiv);

            const timeDiv = document.createElement('div');
            timeDiv.className = 'message-time';
            const displayTimestamp = timestamp || formatTimestampForDisplay();
            timeDiv.textContent = displayTimestamp;
            timeDiv.setAttribute('aria-label', `وقت الرد: ${displayTimestamp}`);
            messageDiv.appendChild(timeDiv);

            chatMessagesContainer.appendChild(messageDiv);
            
            if (save) {
                saveMessageToHistory('assistant', htmlText, displayTimestamp);
            }
            
            scrollToBottom();
            return messageDiv;
            
        } catch (error) {
            console.error("Error displaying assistant message:", error);
            // Display error message to user
            displayErrorMessage("حدث خطأ في عرض الرد");
        }
    }

    /**
     * Creates animated typing indicator
     */
    function createTypingIndicator() {
        const div = document.createElement('div');
        div.className = 'message assistant-message typing-indicator';
        div.setAttribute('aria-label', 'المساعد يكتب');
        
        const logo = document.createElement('img');
        logo.src = "assets/logo.png";
        logo.alt = institutionData ? `${institutionData.name} يكتب` : "المساعد يكتب";
        logo.className = 'assistant-logo';
        logo.onerror = function() {
            this.style.display = 'none';
        };
        
        const content = document.createElement('div');
        content.className = 'message-content';
        content.innerHTML = "جاري الكتابة <div class='loading-dots'><span></span><span></span><span></span></div>";
        
        div.appendChild(logo);
        div.appendChild(content);
        return div;
    }

    /**
     * Displays error message to user
     */
    function displayErrorMessage(errorText) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'message assistant-message error-message';
        errorDiv.setAttribute('role', 'alert');
        errorDiv.setAttribute('aria-label', 'رسالة خطأ');
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content error-content';
        contentDiv.textContent = errorText;
        
        const timeDiv = document.createElement('div');
        timeDiv.className = 'message-time';
        timeDiv.textContent = formatTimestampForDisplay();
        
        errorDiv.appendChild(contentDiv);
        errorDiv.appendChild(timeDiv);
        chatMessagesContainer.appendChild(errorDiv);
        scrollToBottom();
    }

    /**
     * Smooth scroll to bottom of chat
     */
    function scrollToBottom() {
        try {
            chatMessagesContainer.scrollTo({
                top: chatMessagesContainer.scrollHeight,
                behavior: 'smooth'
            });
        } catch (error) {
            // Fallback for browsers that don't support smooth scrolling
            chatMessagesContainer.scrollTop = chatMessagesContainer.scrollHeight;
        }
    }

    // --- Enhanced Send Message Logic ---
    
    /**
     * Validates message before sending
     */
    function validateMessage(message) {
        if (!message || typeof message !== 'string') {
            return { valid: false, error: 'رسالة غير صالحة' };
        }
        
        const trimmed = message.trim();
        if (trimmed.length === 0) {
            return { valid: false, error: 'لا يمكن إرسال رسالة فارغة' };
        }
        
        if (trimmed.length < 2) {
            return { valid: false, error: 'الرسالة قصيرة جداً' };
        }
        
        if (trimmed.length > 2000) {
            return { valid: false, error: 'الرسالة طويلة جداً (الحد الأقصى 2000 حرف)' };
        }
        
        return { valid: true, message: trimmed };
    }

    /**
     * Handles sending message with comprehensive error handling and retry logic
     */
    async function handleSendMessage() {
        // Prevent multiple simultaneous submissions
        if (isFetchingResponse) {
            console.warn("Message submission blocked - already processing");
            return;
        }

        // Throttle message sending
        const now = Date.now();
        if (now - lastMessageTime < MESSAGE_THROTTLE) {
            console.warn("Message throttled - please wait");
            return;
        }
        lastMessageTime = now;

        const messageText = messageInputElement.value;
        
        // Validate message
        const validation = validateMessage(messageText);
        if (!validation.valid) {
            displayErrorMessage(validation.error);
            return;
        }

        const validatedMessage = validation.message;
        
        // Display user message
        displayUserMessage(validatedMessage);
        messageInputElement.value = '';
        messageInputElement.focus();

        // Set loading state
        isFetchingResponse = true;
        sendButtonElement.disabled = true;
        sendButtonElement.setAttribute('aria-label', 'جاري الإرسال...');
        
        const typingIndicatorElement = createTypingIndicator();
        chatMessagesContainer.appendChild(typingIndicatorElement);
        scrollToBottom();

        try {
            await sendMessageWithRetry(validatedMessage, typingIndicatorElement);
        } finally {
            // Always reset state
            isFetchingResponse = false;
            sendButtonElement.disabled = false;
            sendButtonElement.setAttribute('aria-label', 'إرسال الرسالة');
            
            // Remove typing indicator if it still exists
            if (typingIndicatorElement && typingIndicatorElement.parentNode) {
                typingIndicatorElement.remove();
            }
        }
    }

    /**
     * Sends message with retry logic
     */
    async function sendMessageWithRetry(message, typingIndicator) {
        let attempt = 0;
        const maxRetries = MAX_RETRIES;

        while (attempt < maxRetries) {
            try {
                console.info(`Sending message (attempt ${attempt + 1}/${maxRetries})`);
                
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout

                const response = await fetch(CHAT_API_URL, {
                    method: 'POST',
                    headers: { 
                        'Content-Type': 'application/json',
                        'Accept': 'application/json'
                    },
                    body: JSON.stringify({ 
                        message: message, 
                        session_id: currentSessionId 
                    }),
                    signal: controller.signal
                });

                clearTimeout(timeoutId);

                // Remove typing indicator after successful request
                if (typingIndicator && typingIndicator.parentNode) {
                    typingIndicator.remove();
                }

                if (!response.ok) {
                    const errorText = await response.text();
                    let errorDetail = "حدث خطأ أثناء معالجة طلبك.";
                    
                    try {
                        const errorData = JSON.parse(errorText);
                        if (errorData?.detail) {
                            if (Array.isArray(errorData.detail) && errorData.detail[0]?.msg) {
                                errorDetail = errorData.detail[0].msg;
                            } else if (typeof errorData.detail === 'string') {
                                errorDetail = errorData.detail;
                            }
                        }
                    } catch (e) {
                        console.warn("Could not parse error response:", e);
                        errorDetail = `خطأ من الخادم: ${response.status} ${response.statusText}`;
                    }

                    // For client errors (4xx), don't retry
                    if (response.status >= 400 && response.status < 500) {
                        displayAssistantMessage(errorDetail);
                        console.error("Client error:", response.status, errorDetail);
                        return;
                    }

                    // For server errors (5xx), throw to trigger retry
                    throw new Error(`Server error: ${response.status} - ${errorDetail}`);
                }
                
                const data = await response.json();
                
                if (data?.answer) {
                    displayAssistantMessage(data.answer);
                    
                    // Log successful response
                    const processingTime = data.processing_time || 0;
                    const cached = data.cached || false;
                    console.info(`Response received in ${processingTime.toFixed(3)}s (cached: ${cached})`);
                    
                } else {
                    throw new Error("Invalid response format from server");
                }

                return; // Success, exit retry loop

            } catch (error) {
                attempt++;
                console.warn(`Message send failed (attempt ${attempt}):`, error.message);

                if (error.name === 'AbortError') {
                    displayAssistantMessage("انتهت مهلة الاتصال. يرجى المحاولة مرة أخرى.");
                    return;
                }

                if (attempt < maxRetries) {
                    const delayTime = RETRY_DELAY * Math.pow(2, attempt - 1); // Exponential backoff
                    console.info(`Retrying in ${delayTime}ms...`);
                    await delay(delayTime);
                } else {
                    console.error("All retry attempts failed");
                    
                    let userMessage = "لا يمكن الاتصال بالخادم. يرجى التحقق من اتصالك بالإنترنت والمحاولة مرة أخرى.";
                    
                    // More specific error messages
                    if (error.message.includes('fetch')) {
                        userMessage = "فشل في الاتصال بالخادم. يرجى التحقق من اتصالك بالإنترنت.";
                    } else if (error.message.includes('timeout') || error.name === 'AbortError') {
                        userMessage = "انتهت مهلة الاتصال. يرجى المحاولة مرة أخرى.";
                    }
                    
                    displayAssistantMessage(userMessage);
                }
            }
        }
    }

    // --- Enhanced Event Listeners ---
    
    // Throttled send message function
    const throttledSendMessage = throttle(handleSendMessage, MESSAGE_THROTTLE);

    if (sendButtonElement) {
        sendButtonElement.addEventListener('click', throttledSendMessage);
    }

    if (chatFormElement) {
        chatFormElement.addEventListener('submit', function(e) {
            e.preventDefault();
            throttledSendMessage();
        });
    } else if (messageInputElement) {
        messageInputElement.addEventListener('keypress', function (e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                throttledSendMessage();
            }
        });
    }

    // Enhanced quick buttons with better error handling
    document.querySelectorAll('.quick-btn').forEach(button => {
        button.addEventListener('click', () => {
            try {
                const buttonText = button.textContent?.trim();
                if (buttonText && messageInputElement) {
                    messageInputElement.value = buttonText;
                    messageInputElement.focus();
                    throttledSendMessage();
                }
            } catch (error) {
                console.error("Error handling quick button click:", error);
            }
        });
    });

    // Input field enhancements
    if (messageInputElement) {
        // Auto-resize textarea behavior simulation
        messageInputElement.addEventListener('input', function() {
            // Reset height to calculate new height
            this.style.height = 'auto';
            // Set new height (max 100px to prevent excessive growth)
            this.style.height = Math.min(this.scrollHeight, 100) + 'px';
        });

        // Character counter (visual feedback)
        messageInputElement.addEventListener('input', function() {
            const length = this.value.length;
            const maxLength = 2000;
            
            if (length > maxLength * 0.9) { // Warn at 90%
                this.style.borderColor = length > maxLength ? '#ff4444' : '#ff8800';
            } else {
                this.style.borderColor = '';
            }
        });
    }

    // --- Error Recovery and Health Monitoring ---
    
    /**
     * Monitors application health and connectivity
     */
    async function checkApplicationHealth() {
        try {
            const response = await fetch(`${BASE_API_URL}/health`, {
                method: 'GET',
                headers: { 'Accept': 'application/json' }
            });
            
            if (response.ok) {
                const healthData = await response.json();
                console.info("Application health check passed:", healthData.status);
                return true;
            } else {
                console.warn("Health check failed:", response.status);
                return false;
            }
        } catch (error) {
            console.warn("Health check error:", error.message);
            return false;
        }
    }

    // Periodic health monitoring (every 5 minutes)
    setInterval(checkApplicationHealth, 5 * 60 * 1000);

    // --- Page Visibility and Performance ---
    
    // Handle page visibility changes
    document.addEventListener('visibilitychange', function() {
        if (document.visibilityState === 'visible') {
            console.info("Page became visible - checking health");
            checkApplicationHealth();
        }
    });

    // Handle online/offline events
    window.addEventListener('online', function() {
        console.info("Connection restored");
        displayAssistantMessage("تم استعادة الاتصال بالإنترنت.");
    });

    window.addEventListener('offline', function() {
        console.warn("Connection lost");
        displayAssistantMessage("انقطع الاتصال بالإنترنت. يرجى التحقق من اتصالك.");
    });

    // --- Initialization ---
    
    /**
     * Initialize application with comprehensive setup
     */
    async function initializeApplication() {
        try {
            console.info("Initializing AI Gate Application...");
            
            // Initialize session
            initializeSession();
            
            // Load chat history
            loadChatHistory();
            
            // Fetch institution info (with retry logic built-in)
            await fetchAndSetInstitutionInfo();
            
            // Initial health check
            await checkApplicationHealth();
            
            // Focus on input field
            if (messageInputElement) {
                messageInputElement.focus();
            }
            
            console.info("AI Gate Application initialized successfully");
            
        } catch (error) {
            console.error("Application initialization error:", error);
            displayErrorMessage("حدث خطأ أثناء تحميل التطبيق. يرجى إعادة تحميل الصفحة.");
        }
    }

    // Start application initialization
    initializeApplication();

});