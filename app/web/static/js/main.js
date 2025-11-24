document.addEventListener("DOMContentLoaded", () => {
  // ===== Dark Mode Toggle =====
  const themeToggle = document.getElementById("themeToggle");
  const htmlElement = document.documentElement;

  // Check for saved theme preference or default to light mode
  const currentTheme = localStorage.getItem("theme") || "light";
  if (currentTheme === "dark") {
    htmlElement.classList.add("dark-mode");
  }

  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      htmlElement.classList.toggle("dark-mode");
      const theme = htmlElement.classList.contains("dark-mode") ? "dark" : "light";
      localStorage.setItem("theme", theme);
    });
  }

  const searchInput = document.querySelector("form.search-form input[type='text']");
  const searchForm = document.querySelector("form.search-form");

  if (!searchInput) {
    return;
  }

  // RTL direction detection for Arabic
  const applyDirection = () => {
    const value = searchInput.value || "";
    const hasArabic = /[\u0600-\u06FF]/.test(value);
    searchInput.dir = hasArabic ? "rtl" : "ltr";
  };

  searchInput.addEventListener("input", applyDirection);
  applyDirection();

  // Auto-focus search input if no query
  if (!searchInput.value) {
    searchInput.focus();
  }

  // Loading state on form submit
  if (searchForm) {
    searchForm.addEventListener("submit", () => {
      const button = searchForm.querySelector("button[type='submit']");
      const skeletonContainer = document.getElementById("skeletonContainer");
      if (button && searchInput.value.trim()) {
        button.innerHTML = '<span class="spinner"></span>';
        button.disabled = true;
        // Show skeleton loading screens
        if (skeletonContainer) {
          skeletonContainer.hidden = false;
        }
      }
    });
  }

  // Keyboard shortcut: "/" to focus search
  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && !e.ctrlKey && !e.metaKey && !e.altKey) {
      if (searchInput && document.activeElement !== searchInput) {
        e.preventDefault();
        searchInput.focus();
        searchInput.select();
      }
    }
  });

  // ===== Clear Button Functionality =====
  const clearBtn = document.getElementById("clearBtn");

  // Show/hide clear button based on input value
  const updateClearButton = () => {
    if (clearBtn && searchInput) {
      clearBtn.hidden = !searchInput.value.trim();
    }
  };

  // Update clear button on page load
  updateClearButton();

  // Update clear button as user types
  if (searchInput) {
    searchInput.addEventListener("input", updateClearButton);
  }

  // Clear input when X button is clicked
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      if (searchInput) {
        searchInput.value = "";
        searchInput.focus();
        updateClearButton();
        applyDirection();
      }
    });
  }

  // ===== Voice Search Button =====
  const voiceBtn = document.getElementById("voiceBtn");
  let mediaRecorder = null;
  let mediaStream = null;
  let audioChunks = [];
  let voiceState = "idle";

  const cleanupMedia = () => {
    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
    }
    mediaRecorder = null;
    mediaStream = null;
    audioChunks = [];
  };

  const setVoiceState = (state) => {
    voiceState = state;
    if (!voiceBtn) {
      return;
    }
    voiceBtn.dataset.state = state;
    const shouldDisable = state === "requesting" || state === "uploading";
    voiceBtn.disabled = shouldDisable;
    if (state === "recording") {
      voiceBtn.setAttribute("aria-pressed", "true");
      voiceBtn.setAttribute("aria-label", "Stop recording");
    } else {
      voiceBtn.setAttribute("aria-pressed", "false");
      voiceBtn.setAttribute("aria-label", "Start voice search");
      if (state === "idle") {
        voiceBtn.disabled = false;
      }
    }
  };

  const submitSearch = () => {
    if (!searchForm || !searchInput) {
      return;
    }
    updateClearButton();
    applyDirection();
    if (searchInput.value.trim()) {
      if (typeof searchForm.requestSubmit === "function") {
        searchForm.requestSubmit();
      } else {
        searchForm.submit();
      }
    }
  };

  const uploadRecording = async (blob) => {
    if (!blob || !voiceBtn) {
      return;
    }
    setVoiceState("uploading");
    const formData = new FormData();
    formData.append("audio", blob, `voice-${Date.now()}.webm`);

    try {
      const response = await fetch("/transcribe", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();

      if (response.ok && payload.success && payload.text) {
        if (searchInput) {
          searchInput.value = payload.text;
          searchInput.focus();
        }
        submitSearch();
      } else {
        showToast(payload.error || "Could not transcribe audio.");
      }
    } catch (err) {
      console.error("Transcription failed:", err);
      showToast("Voice transcription failed. Please try again.");
    } finally {
      cleanupMedia();
      setVoiceState("idle");
    }
  };

  const handleRecordingStop = async () => {
    if (!audioChunks.length) {
      cleanupMedia();
      setVoiceState("idle");
      return;
    }
    const mimeType = mediaRecorder?.mimeType || "audio/webm";
    const blob = new Blob(audioChunks, { type: mimeType });
    await uploadRecording(blob);
  };

  const stopRecording = () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    } else {
      cleanupMedia();
      setVoiceState("idle");
    }
  };

  const startRecording = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showToast("Your browser does not support voice search.");
      return;
    }
    setVoiceState("requesting");
    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(mediaStream);
      audioChunks = [];
      mediaRecorder.addEventListener("dataavailable", (event) => {
        if (event.data?.size) {
          audioChunks.push(event.data);
        }
      });
      mediaRecorder.addEventListener("stop", handleRecordingStop);
      mediaRecorder.start();
      setVoiceState("recording");
    } catch (err) {
      console.error("Microphone permission denied", err);
      cleanupMedia();
      setVoiceState("idle");
      showToast("Microphone access denied.");
    }
  };

  if (voiceBtn) {
    setVoiceState("idle");
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      voiceBtn.disabled = true;
      voiceBtn.title = "Voice search is not supported in this browser.";
    } else {
      voiceBtn.addEventListener("click", () => {
        if (voiceState === "recording") {
          stopRecording();
        } else if (voiceState === "idle") {
          startRecording();
        }
      });
    }
  }

  // ===== Upload Feature =====
  const uploadBtn = document.getElementById("uploadBtn");
  const uploadModal = document.getElementById("uploadModal");
  const uploadBackdrop = document.getElementById("uploadBackdrop");
  const closeModal = document.getElementById("closeModal");
  const cancelBtn = document.getElementById("cancelBtn");
  const uploadForm = document.getElementById("uploadForm");
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const browseBtn = document.getElementById("browseBtn");
  const fileInfo = document.getElementById("fileInfo");
  const fileName = document.getElementById("fileName");
  const removeFile = document.getElementById("removeFile");
  const submitBtn = document.getElementById("submitBtn");
  const uploadProgress = document.getElementById("uploadProgress");
  const progressBar = document.getElementById("progressBar");
  const progressText = document.getElementById("progressText");
  const toast = document.getElementById("toast");
  const toastMessage = document.getElementById("toastMessage");

  let selectedFile = null;

  // Show toast notification
  const showToast = (message, duration = 4000) => {
    if (toastMessage) {
      toastMessage.textContent = message;
    }
    if (toast) {
      toast.hidden = false;
      setTimeout(() => {
        toast.hidden = true;
      }, duration);
    }
  };

  // Open modal
  const openModal = () => {
    if (uploadModal) {
      uploadModal.hidden = false;
    }
  };

  // Close modal
  const closeModalHandler = () => {
    if (uploadModal) {
      uploadModal.hidden = true;
    }
    resetForm();
  };

  // Reset form
  const resetForm = () => {
    selectedFile = null;
    if (fileInput) {
      fileInput.value = "";
    }
    if (fileInfo) {
      fileInfo.hidden = true;
    }
    if (dropzone) {
      dropzone.hidden = false;
    }
    if (submitBtn) {
      submitBtn.disabled = true;
    }
    if (uploadProgress) {
      uploadProgress.hidden = true;
    }
  };

  // Validate file
  const validateFile = (file) => {
    const maxSize = 50 * 1024 * 1024; // 50MB
    const allowedExtensions = new Set(["pdf", "mp3", "wav", "m4a", "mp4", "flac"]);

    if (!file) {
      return { valid: false, error: "No file selected" };
    }

    const nameParts = file.name?.split(".") || [];
    const extension = nameParts.length > 1 ? nameParts.pop().toLowerCase() : "";
    const isAudioMime = file.type.startsWith("audio/");
    const isSupported =
      allowedExtensions.has(extension) ||
      file.type === "application/pdf" ||
      isAudioMime ||
      file.type === "video/mp4";

    if (!isSupported) {
      return {
        valid: false,
        error: "Invalid file type. Upload PDF or audio files (MP3, WAV, M4A, MP4, FLAC).",
      };
    }

    if (file.size > maxSize) {
      return { valid: false, error: "File too large. Maximum size is 50MB." };
    }

    return { valid: true };
  };

  // Handle file selection
  const handleFileSelect = (file) => {
    const validation = validateFile(file);

    if (!validation.valid) {
      showToast(validation.error);
      return;
    }

    selectedFile = file;

    if (fileName) {
      fileName.textContent = file.name;
    }
    if (fileInfo) {
      fileInfo.hidden = false;
    }
    if (dropzone) {
      dropzone.hidden = true;
    }
    if (submitBtn) {
      submitBtn.disabled = false;
    }
  };

  // Event listeners
  if (uploadBtn) {
    uploadBtn.addEventListener("click", openModal);
  }

  if (closeModal) {
    closeModal.addEventListener("click", closeModalHandler);
  }

  if (uploadBackdrop) {
    uploadBackdrop.addEventListener("click", closeModalHandler);
  }

  if (cancelBtn) {
    cancelBtn.addEventListener("click", closeModalHandler);
  }

  if (browseBtn) {
    browseBtn.addEventListener("click", () => {
      if (fileInput) {
        fileInput.click();
      }
    });
  }

  if (fileInput) {
    fileInput.addEventListener("change", (e) => {
      const file = e.target.files?.[0];
      if (file) {
        handleFileSelect(file);
      }
    });
  }

  if (removeFile) {
    removeFile.addEventListener("click", () => {
      resetForm();
    });
  }

  // Drag and drop
  if (dropzone) {
    dropzone.addEventListener("click", () => {
      if (fileInput) {
        fileInput.click();
      }
    });

    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("drag-over");
    });

    dropzone.addEventListener("dragleave", () => {
      dropzone.classList.remove("drag-over");
    });

    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag-over");

      const file = e.dataTransfer?.files?.[0];
      if (file) {
        handleFileSelect(file);
      }
    });
  }

  // Form submit
  if (uploadForm) {
    uploadForm.addEventListener("submit", async (e) => {
      e.preventDefault();

      if (!selectedFile) {
        showToast("Please select a file");
        return;
      }

      const languageSelect = document.getElementById("languageSelect");
      const language = languageSelect?.value || "auto";

      // Show progress
      if (uploadProgress) {
        uploadProgress.hidden = false;
      }
      if (submitBtn) {
        submitBtn.disabled = true;
      }
      if (cancelBtn) {
        cancelBtn.disabled = true;
      }

      try {
        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("language", language);

        const response = await fetch("/upload", {
          method: "POST",
          body: formData,
        });

        const result = await response.json();

        if (response.ok && result.success) {
          showToast(`Added ${result.chunks || 0} chunks from ${result.filename || "document"}`);
          closeModalHandler();

          // Optionally reload page to show updated document count
          setTimeout(() => {
            window.location.reload();
          }, 2000);
        } else {
          showToast(result.error || "Upload failed. Please try again.");
          if (uploadProgress) {
            uploadProgress.hidden = true;
          }
          if (submitBtn) {
            submitBtn.disabled = false;
          }
          if (cancelBtn) {
            cancelBtn.disabled = false;
          }
        }
      } catch (error) {
        console.error("Upload error:", error);
        showToast("Upload failed. Please try again.");
        if (uploadProgress) {
          uploadProgress.hidden = true;
        }
        if (submitBtn) {
          submitBtn.disabled = false;
        }
        if (cancelBtn) {
          cancelBtn.disabled = false;
        }
      }
    });
  }

  // ===== Filter and Sort Functionality =====
  const languageFilter = document.getElementById("languageFilter");
  const docTypeFilter = document.getElementById("docTypeFilter");
  const sortBy = document.getElementById("sortBy");
  const resetFilters = document.getElementById("resetFilters");
  const cardsContainer = document.getElementById("cardsContainer");

  if (languageFilter && docTypeFilter && sortBy && cardsContainer) {
    const applyFilters = () => {
      const cards = Array.from(cardsContainer.querySelectorAll(".card"));
      const languageValue = languageFilter.value;
      const docTypeValue = docTypeFilter.value;
      const sortValue = sortBy.value;

      // Filter cards
      cards.forEach(card => {
        const cardLanguage = card.getAttribute("data-language");
        const cardDocType = card.getAttribute("data-doc-type");

        const languageMatch = languageValue === "all" || cardLanguage === languageValue;
        const docTypeMatch = docTypeValue === "all" || cardDocType === docTypeValue;

        card.hidden = !(languageMatch && docTypeMatch);
      });

      // Sort visible cards
      const visibleCards = cards.filter(card => !card.hidden);

      if (sortValue === "type") {
        visibleCards.sort((a, b) => {
          const typeA = a.getAttribute("data-doc-type");
          const typeB = b.getAttribute("data-doc-type");
          return typeA.localeCompare(typeB);
        });
      } else if (sortValue === "language") {
        visibleCards.sort((a, b) => {
          const langA = a.getAttribute("data-language");
          const langB = b.getAttribute("data-language");
          return langA.localeCompare(langB);
        });
      } else {
        // Sort by original relevance order
        visibleCards.sort((a, b) => {
          const indexA = parseInt(a.getAttribute("data-original-index"));
          const indexB = parseInt(b.getAttribute("data-original-index"));
          return indexA - indexB;
        });
      }

      // Reorder cards in DOM
      visibleCards.forEach(card => cardsContainer.appendChild(card));
    };

    languageFilter.addEventListener("change", applyFilters);
    docTypeFilter.addEventListener("change", applyFilters);
    sortBy.addEventListener("change", applyFilters);

    if (resetFilters) {
      resetFilters.addEventListener("click", () => {
        languageFilter.value = "all";
        docTypeFilter.value = "all";
        sortBy.value = "relevance";
        applyFilters();
      });
    }
  }
});
