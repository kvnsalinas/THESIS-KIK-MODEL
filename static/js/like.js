document.addEventListener('DOMContentLoaded', function() {
    const likeButtons = document.querySelectorAll('.btn-like');
    
    likeButtons.forEach(button => {
        button.addEventListener('click', function() {
            const placeName = this.getAttribute('data-place');
            
            fetch(`/like_place/${encodeURIComponent(placeName)}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    button.innerHTML = '<i class="fas fa-heart"></i> Liked!';
                    button.classList.add('liked');
                    button.disabled = true;
                    
                    const notification = document.getElementById('notification');
                    notification.querySelector('span').textContent = data.message;
                    notification.classList.add('show');
                    setTimeout(() => notification.classList.remove('show'), 3000);
                }
            })
            .catch(error => {
                console.error('Error:', error);
                const notification = document.getElementById('notification');
                notification.querySelector('span').textContent = 'Error adding to favorites';
                notification.classList.add('show');
                setTimeout(() => notification.classList.remove('show'), 3000);
            });
        });
    });
});